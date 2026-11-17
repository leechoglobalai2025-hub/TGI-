#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 新盲测 D（2026-11-16）。规则与判定标准预先写在 SESSION_20261116_TGI_Weights.md。

不使用 MIT 的层号和神经元编号。全部用没用过的题目（题库第 201–264 题）。
  1. collect：未修改模型在第 201–232 题上生成（上限 8192），记录循环态 / 正常态的激活和残差。
  2. 选层：Astra 判据 S(l) = d_l · W_l (循环态平均激活 − 正常态平均激活)，d_l 为第 l 层输出处的单位幽灵方向；
     排除最后一层，取 S 最大的 2 层。
  3. 选神经元：这两层各取写入量（循环态平均激活 × (d_l · 输出列)）最大的 64 个。
  4. 干预：把选中的神经元置零（论文 §1.3）。
     （最初写的是减方向；2026-11-16 C 测试显示我们的减方向在单层上不显著，用户决定改成置零。改动发生在 D 运行之前。）
  5. 评测：第 233–264 题，同一随机种子，四个条件：不改动 / 盲测 / 对照（同层随机 64 个神经元置零）/ 参照（MIT 的实际改动）。

用法：python tgi_blind2.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tgi_rcs as R  # noqa: E402

OUT = R.ROOT / "data" / "rcs_blind2_20261116"
MODEL = os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B")
COLLECT = (200, 232)        # 题库下标（从 0 起），即第 201–232 题
EVAL = (232, 264)           # 第 233–264 题
CAP, BATCH, SEED, K = 8192, 32, 42, 64
log = R.log


def select(model, z):
    """返回 (选出的两层, {层: 神经元}, {层: 单位幽灵方向}, 各层 S)。只用 collect 的数据和模型自身权重。"""
    L = len(model.model.layers)
    dh = z["rep_h"] - z["new_h"]
    da = z["rep_mean"] - z["new_mean"]
    S, writer, dirs = np.zeros(L), [], []
    for l, layer in enumerate(model.model.layers):
        W = layer.mlp.down_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        d = dh[l] / (np.linalg.norm(dh[l]) + 1e-12)
        S[l] = d @ (W @ da[l])
        writer.append(z["rep_mean"][l] * (d @ W))
        dirs.append(d)
    order = [int(l) for l in np.argsort(S[:L - 1])[::-1]]            # 排除最后一层
    layers = order[:2]
    cols = {l: [int(c) for c in np.argsort(writer[l])[::-1][:K]] for l in layers}
    return layers, cols, {l: dirs[l] for l in layers}, S, order


class SubtractDirection:
    """从指定神经元的输出列里减去方向 v 上的分量；restore() 恢复原权重。"""

    def __init__(self, model, cols, dirs):
        self.saved = {}
        for l, c in cols.items():
            p = model.model.layers[l].mlp.down_proj.weight
            self.saved[l] = (p, p.data.clone())
            W = p.data.float()
            v = torch.tensor(dirs[l], dtype=torch.float32, device=W.device)
            idx = torch.tensor(c, device=W.device)
            sub = W[:, idx]
            before = (v @ sub).abs().mean().item()
            W[:, idx] = sub - torch.outer(v, v @ sub)
            p.data.copy_(W.to(p.dtype))
            after = (v @ p.data.float()[:, idx]).abs().mean().item()
            mask = torch.ones(W.shape[1], dtype=torch.bool, device=W.device)
            mask[idx] = False
            untouched = torch.equal(p.data[:, mask], self.saved[l][1][:, mask])
            log(f"   第 {l} 层：选中列在方向上的平均分量 {before:.4f} → {after:.4f}；其余列未变 = {untouched}")

    def restore(self):
        for p, w in self.saved.values():
            p.data.copy_(w)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, 24 / total))
    model, tok = R.load_model(MODEL)
    problems = [json.loads(x) for x in open(R.REF / "math_train.jsonl", encoding="utf-8")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # 1. collect
    t0 = time.time()
    ids_file = OUT / "collect_ids.pt"
    if ids_file.exists():
        outs = torch.load(ids_file)
        log(f"沿用已保存的生成结果（{len(outs)} 条）")
    else:
        outs = R.generate(model, tok, problems[COLLECT[0]:COLLECT[1]], CAP, BATCH, SEED)
        torch.save([{k: o[k] for k in ("id", "prompt_ids", "new_ids", "ended")} for o in outs], ids_file)
    m = R.metrics(tok, outs, CAP)
    log(f"[采集：未修改模型，第 {COLLECT[0] + 1}–{COLLECT[1]} 题] {R.fmt(m)}  ({time.time() - t0:.0f}s)")
    if not (OUT / "collect.npz").exists():
        act = R.collect_activations(model, outs, pad)
        np.savez_compressed(OUT / "collect.npz", rep_mean=act["rep"]["mean"], rep_var=act["rep"]["var"],
                            new_mean=act["new"]["mean"], new_var=act["new"]["var"],
                            rep_h=act["rep"]["h"], new_h=act["new"]["h"],
                            n=np.array([act["rep"]["n"], act["new"]["n"]]))
    z = np.load(OUT / "collect.npz")
    log(f"激活统计：循环内 token {int(z['n'][0]):,} 个，正常 token {int(z['n'][1]):,} 个")

    # 2–3. 选层、选神经元
    layers, cols, dirs, S, order = select(model, z)
    I = model.model.layers[0].mlp.down_proj.in_features
    mit = R.mit_columns(I)
    log("各层 S（从大到小，不含最后一层）：" + " ".join(f"L{l}:{S[l]:.3f}" for l in order[:8])
        + f"   最后一层 L{len(S) - 1}:{S[-1]:.3f}")
    log(f"盲测选出的层：{layers}")
    overlap = {l: len(set(cols[l]) & set(mit.get(l, []))) for l in layers}
    log(f"（选定之后才对照）与 MIT 改动的层 {sorted(mit)} 比较；选中神经元与 MIT 的重合：{overlap}")
    rng = np.random.default_rng(SEED + 11)
    rand_cols = {l: [int(c) for c in rng.choice(I, K, replace=False)] for l in layers}
    sel = {"layers": layers, "cols": {str(l): cols[l] for l in layers}, "rand_cols": {str(l): rand_cols[l] for l in layers},
           "S": S.tolist(), "order_excl_last": order, "mit_overlap": {str(l): overlap[l] for l in layers},
           "collect_metrics": m}
    (OUT / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), "utf-8")
    np.savez_compressed(OUT / "dirs.npz", **{f"L{l}": dirs[l] for l in layers})

    # 5. 评测
    held = problems[EVAL[0]:EVAL[1]]
    res_file = OUT / "ablate.json"
    results = json.loads(res_file.read_text("utf-8")) if res_file.exists() else {}

    def run(name, setup):
        if name in results:
            log(f"[{name}] 已有结果，跳过")
            return
        t = time.time()
        undo = setup()
        outs_c = R.generate(model, tok, held, CAP, BATCH, SEED + 1000)
        undo()
        results[name] = R.metrics(tok, outs_c, CAP)
        torch.save([{k: o[k] for k in ("id", "new_ids", "ended")} for o in outs_c], OUT / f"gen_{len(results)}.pt")
        log(f"[{name}] {R.fmt(results[name])}  ({time.time() - t:.0f}s)")
        res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")

    def setup_zero(c):
        def f():
            h = R.ZeroNeurons(model, c)
            return h.remove
        return f

    def setup_mit():
        saved = {}
        for l, w in R.mit_edited_weights(model).items():
            p = model.model.layers[l].mlp.down_proj.weight
            saved[l] = (p, p.data.clone())
            p.data.copy_(w.to(p.device))

        def undo():
            for p, w in saved.values():
                p.data.copy_(w)
        return undo

    run("不改动", lambda: (lambda: None))
    run("盲测：自选层自选 64 置零", setup_zero(cols))
    run("对照：同层随机 64 置零", setup_zero(rand_cols))
    run("参照：MIT 的实际改动", setup_mit)
    log(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB · 输出 {OUT}")


if __name__ == "__main__":
    main()
