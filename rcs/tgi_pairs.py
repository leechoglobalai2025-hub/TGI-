#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 测试 E（2026-11-16）：因果验证的快速检验 —— 成对文本 + 对数概率（仿 MIT/Astra 的做法）。
内容与判定标准预先写在 SESSION_20261116_TGI_Weights.md。

文本直接取自 Astra 公开的工作记录（Ref/mit_wsp_exp8/p50c_logs/tool_calls.jsonl）：
  选择用的 4 对（第 8 号调用）；迁移检验用的 6 对（第 58 号调用的前 12 条）。偶数下标 = 正常，奇数下标 = 重复。

  E1  从 MIT 的差分还原方向 u_MIT 和剂量；与我们的方向的余弦。
  E2  用 4 对文本按 Astra 的做法重算方向（最后 36 个 token 的残差均值，成对相减），与 u_MIT 的余弦。
  E3  对数概率检验：6 对迁移文本，分叉处到结尾的平均对数概率，改动后 − 改动前。

改动公式（与 MIT 相同）：W = W0 − α·outer(u, mask·(uᵀW0))。α=1 即"减去方向"，MIT 最终用 α=6。

用法：python tgi_pairs.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tgi_rcs as R  # noqa: E402

OUT = R.ROOT / "data" / "pairs_20261116"
MODEL = os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B")
V2 = R.ROOT / "data" / "rcs_mit_Qwen3.5-0.8B_v2"
B2 = R.ROOT / "data" / "rcs_blind2_20261116"
LAYERS = (15, 19)
log = R.log


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def load_texts():
    rows = [json.loads(x) for x in open(R.REF / "p50c_logs" / "tool_calls.jsonl", encoding="utf-8")]
    return rows[8]["request"]["texts"], rows[58]["request"]["texts"][:12]


@torch.no_grad()
def forward(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    out = model(input_ids=ids, output_hidden_states=True)
    lp = torch.log_softmax(out.logits[0].float(), -1)                    # [T, V]
    tok_lp = lp[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()     # tok_lp[j] = 第 j+1 个 token 的对数概率
    hs = {l: out.hidden_states[l + 1][0].float().cpu().numpy() for l in LAYERS}   # 第 l 层的输出
    return ids[0].tolist(), tok_lp, hs


class Edit:
    """W = W0 − α·outer(u, mask·(uᵀW0))；或把 mask 内的列置零；或直接换成给定权重。"""

    def __init__(self, model):
        self.model, self.saved = model, {}

    def _w(self, l):
        p = self.model.model.layers[l].mlp.down_proj.weight
        if l not in self.saved:
            self.saved[l] = p.data.clone()
        return p

    def direction(self, dirs, cols, alpha):
        rel = {}
        for l in LAYERS:
            p = self._w(l)
            W = p.data.float()
            u = torch.tensor(unit(dirs[l]), dtype=torch.float32, device=W.device)
            mask = torch.zeros(W.shape[1], device=W.device)
            mask[torch.tensor(cols[l], device=W.device)] = 1
            new = W - alpha * torch.outer(u, mask * (u @ W))
            rel[l] = ((new - W).norm() / W.norm()).item()
            p.data.copy_(new.to(p.dtype))
        return rel

    def zero(self, cols):
        for l in LAYERS:
            p = self._w(l)
            p.data[:, torch.tensor(cols[l], device=p.device)] = 0

    def set(self, weights):
        for l, w in weights.items():
            self._w(l).data.copy_(w.to(self.saved[l].device))

    def restore(self):
        for l, w in self.saved.items():
            self.model.model.layers[l].mlp.down_proj.weight.data.copy_(w)
        self.saved = {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = R.load_model(MODEL)
    sel_texts, tr_texts = load_texts()
    I = model.model.layers[0].mlp.down_proj.in_features
    res = {}

    # ═══ E1：还原 MIT 的方向和剂量 ═══
    log("══ E1：MIT 改动的方向与剂量 ══")
    mit_cols = R.mit_columns(I)
    mit_w = R.mit_edited_weights(model)
    u_mit, res["E1"] = {}, {}
    for l in LAYERS:
        W0 = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu().double()
        dW = mit_w[l].float().double() - W0
        U, S, Vt = torch.linalg.svd(dW[:, mit_cols[l]], full_matrices=False)
        u = U[:, 0].numpy()
        c0 = torch.tensor(u) @ W0[:, mit_cols[l]]
        c1 = torch.tensor(u) @ (W0 + dW)[:, mit_cols[l]]
        alpha = float(((c0 - c1) / c0).median())
        u_mit[l] = u
        res["E1"][str(l)] = {"rank1_energy": float(S[0] ** 2 / (S ** 2).sum()), "alpha_median": alpha,
                             "rel_change": float(dW.norm() / W0.norm())}
        log(f"  第 {l} 层：改动的第一奇异值占能量 {res['E1'][str(l)]['rank1_energy']:.4f}；"
            f"剂量 α 中位数 {alpha:.2f}；相对改动 {res['E1'][str(l)]['rel_change']:.4f}")
    z = np.load(V2 / "collect.npz")
    v_y = {l: unit(z["rep_h"][l] - z["new_h"][l]) for l in LAYERS}                     # 昨天（第 1–32 题）
    d2 = np.load(B2 / "dirs.npz")
    v_t = {l: unit(d2[f"L{l}"]) for l in LAYERS}                                        # 今天盲测（第 201–232 题）
    for l in LAYERS:
        if u_mit[l] @ v_y[l] < 0:
            u_mit[l] = -u_mit[l]                                                        # 奇异向量符号任意，统一到同向
        cy, ct, cyt = float(u_mit[l] @ v_y[l]), float(u_mit[l] @ v_t[l]), float(v_y[l] @ v_t[l])
        res["E1"][str(l)].update({"cos_mit_ours_yesterday": cy, "cos_mit_ours_today": ct, "cos_ours_y_t": cyt})
        log(f"  第 {l} 层余弦：u_MIT·我们昨天的 {cy:.3f}；u_MIT·我们今天的 {ct:.3f}；我们昨天·今天 {cyt:.3f}")

    # ═══ E2：按 Astra 的做法重算方向 ═══
    log("══ E2：用 Astra 的 4 对文本重算方向 ══")
    tails = {l: [] for l in LAYERS}
    for t in sel_texts:
        ids, _, hs = forward(model, tok, t)
        for l in LAYERS:
            tails[l].append(hs[l][len(ids) - 36:len(ids)].mean(0))
    u_pair, res["E2"] = {}, {}
    for l in LAYERS:
        m = np.stack(tails[l])
        u_pair[l] = unit((m[1::2] - m[::2]).mean(0))
        c = float(u_pair[l] @ u_mit[l])
        res["E2"][str(l)] = {"cos_pair_mit": c, "cos_pair_ours_yesterday": float(u_pair[l] @ v_y[l]),
                             "cos_pair_ours_today": float(u_pair[l] @ v_t[l])}
        log(f"  第 {l} 层：重算方向·u_MIT = {c:.4f}（Astra 自报约 0.995）；重算方向·我们昨天的 "
            f"{u_pair[l] @ v_y[l]:.3f}；·我们今天的 {u_pair[l] @ v_t[l]:.3f}")

    # ═══ E3：对数概率检验 ═══
    log("══ E3：对数概率检验（6 对迁移文本；分叉处到结尾的平均对数概率，改动后 − 改动前）══")
    enc = [tok(t, add_special_tokens=False).input_ids for t in tr_texts]
    branch = []
    for i in range(0, 12, 2):
        a, b = enc[i], enc[i + 1]
        k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
        branch += [k, k]

    def probe():
        vals = []
        for i, t in enumerate(tr_texts):
            _, tok_lp, _ = forward(model, tok, t)
            vals.append(float(tok_lp[branch[i] - 1:].mean()))             # 第 branch..末尾 个 token
        v = np.array(vals)
        return v[::2], v[1::2]                                            # 正常，重复

    base_p, base_r = probe()
    res["E3"] = {"base": {"productive": float(base_p.mean()), "repetitive": float(base_r.mean())}, "conds": {}}
    log(f"  改动前：正常段 {base_p.mean():.4f}，重复段 {base_r.mean():.4f}（Astra 自报 −1.673 / −0.685）")
    writer = np.load(V2 / "rank_scores.npz")["writer"]
    cols_y = {l: [int(c) for c in np.argsort(writer[l])[::-1][:64]] for l in LAYERS}
    sel = json.loads((B2 / "selection.json").read_text("utf-8"))
    cols_t = {l: sel["cols"][str(l)] for l in LAYERS}
    rng = np.random.default_rng(2026)
    u_rand = {}
    for l in LAYERS:                                                      # 与我们的方向正交的随机单位方向
        r = rng.standard_normal(len(v_y[l]))
        u_rand[l] = unit(r - (r @ v_y[l]) * v_y[l])
    ed = Edit(model)

    def run(name, fn):
        info = fn()
        p, r = probe()
        ed.restore()
        dp, dr = float((p - base_p).mean()), float((r - base_r).mean())
        res["E3"]["conds"][name] = {"d_productive": dp, "d_repetitive": dr, "per_case_rep": (r - base_r).tolist(),
                                    "per_case_prod": (p - base_p).tolist(), "info": info}
        extra = f"   相对改动 {info}" if isinstance(info, dict) else ""
        log(f"  {name:<34} 重复段 {dr:+.4f}   正常段 {dp:+.4f}   重复段下降的对数 {sum(x < 0 for x in (r - base_r))}/6{extra}")

    run("校准：MIT 的实际改动", lambda: ed.set(mit_w))
    for a in (1, 2, 4, 6):
        run(f"我们的方向+我们的神经元（昨天） α={a}", lambda a=a: ed.direction(v_y, cols_y, a))
    for a in (1, 6):
        run(f"今天盲测的方向+神经元 α={a}", lambda a=a: ed.direction(v_t, cols_t, a))
    run("交叉：我们的神经元+MIT 的方向 α=6", lambda: ed.direction(u_mit, cols_y, 6))
    run("交叉：MIT 的神经元+我们的方向 α=6", lambda: ed.direction(v_y, mit_cols, 6))
    run("重算的成对方向+我们的神经元 α=6", lambda: ed.direction(u_pair, cols_y, 6))
    run("置零：我们的 128 个（昨天）", lambda: ed.zero(cols_y))
    run("置零：MIT 的 128 个", lambda: ed.zero(mit_cols))

    def rand_matched():                                                   # 同范数：缩放 α 使改动范数与"我们的方向 α=6"相同
        target = ed.direction(v_y, cols_y, 6)
        ed.restore()
        one = ed.direction(u_rand, cols_y, 1)
        ed.restore()
        info = {}
        for l in LAYERS:
            p = ed._w(l)
            W = p.data.float()
            u = torch.tensor(u_rand[l], dtype=torch.float32, device=W.device)
            mask = torch.zeros(W.shape[1], device=W.device)
            mask[torch.tensor(cols_y[l], device=W.device)] = 1
            a = target[l] / max(one[l], 1e-12)
            new = W - a * torch.outer(u, mask * (u @ W))
            info[l] = ((new - W).norm() / W.norm()).item()
            p.data.copy_(new.to(p.dtype))
        return info
    run("对照：我们的神经元+同范数随机正交方向", rand_matched)

    (OUT / "pairs.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    np.savez_compressed(OUT / "directions.npz", **{f"mit_L{l}": u_mit[l] for l in LAYERS},
                        **{f"pair_L{l}": u_pair[l] for l in LAYERS})
    log(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB · 输出 {OUT}")


if __name__ == "__main__":
    main()
