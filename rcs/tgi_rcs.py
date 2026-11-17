#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI RCS 检验 — 用 MIT "Weight-Space Programming" 实验 8 的公开改动当标准答案，
检验 TGI 的生成元定位方法能否独立找到那 128 个 MLP 神经元。

背景：MIT 让 GPT-6 Astra 修改 Qwen3.5-0.8B 第 15、19 层 MLP down_proj 的各 64 列，
消除思考模式下的循环重复。改动文件公开在 github.com/RikiyaT/lm_understand-logs（exp8_math/runs/p50c/edit）。

三个阶段（预注册预测见 SESSION_20261115_TGI_Probe.md）：
  collect  在未修改的模型上复现循环（思考模式，MIT 的采样设置，token 上限压低），
           再把生成结果做一次不采样的前向，记录每个 MLP 神经元在"重复 token"和"新 token"上的激活。
  rank     P1 按领域的候选集（需要先在同一模型上跑 tgi_probe.py）与 MIT 神经元的重合；
           P2 "重复 − 新"激活差排序与 MIT 神经元的重合；
           附加（事后探索，未预注册）：按"写入幽灵方向的量"排序。
  ablate   P3 RCS 分组消融（论文 §1.3：置零神经元后重新生成，看重复率变化）。

实现时的选择（论文没有规定，需用户审定）：
  a) "重复 token"：以该 token 结尾的 8 个连续 token 在本次生成的更早位置出现过。
  b) 激活差得分 = (重复均值 − 新均值) / sqrt((两边方差之和)/2)。
  c) RepRate 沿用 tgi_app.repetition_RR 的定义（按空格切词，1 − 不同词数/总词数），未改动。
  d) token 上限远低于 MIT 的 32768（本机时间有限），所以"撑满上限比例"不能和 MIT 的数字直接比，
     只在本实验的各条件之间比较。

用法：
  python tgi_rcs.py --phase collect
  python tgi_rcs.py --phase rank --probe_dir <tgi_probe 在同一模型上的输出目录>
  python tgi_rcs.py --phase ablate
"""
import argparse
import hashlib
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TGI_RCS_ROOT", str(Path(__file__).resolve().parent.parent)))
REF = ROOT / "Ref" / "mit_wsp_exp8"           # MIT 公开文件的本地副本（edit.npz、manifest.json、math_train.jsonl）
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
NGRAM = 8
EDIT_LAYERS = (15, 19)


def log(msg):
    print(msg, flush=True)


# ─── 模型与生成 ──────────────────────────────────────────────────────────
def load_model(path):
    try:  # optional local speed-up on some unified-memory hosts
        import gb10_fastload
        gb10_fastload.apply()
    except Exception:
        pass
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def make_presence(pen, n_prompt):
    """与 vLLM 的 presence_penalty 相同：已生成过的 token，logit 减去 pen（只减一次）。"""
    from transformers import LogitsProcessor

    class Presence(LogitsProcessor):
        def __call__(self, ids, scores):
            gen = ids[:, n_prompt:]
            if gen.shape[1] == 0:
                return scores
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, gen, True)
            return scores - pen * mask
    return Presence()


@torch.no_grad()
def generate(model, tok, problems, cap, batch, seed):
    """MIT 的评测设置：思考开，temperature 0.6，top_p 0.95，top_k 20，presence penalty 1.5。"""
    from transformers import LogitsProcessorList
    stop = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.padding_side = "left"
    outs = []
    for k in range(0, len(problems), batch):
        chunk = problems[k:k + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        n_prompt = enc.input_ids.shape[1]
        torch.manual_seed(seed + k)
        out = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                             logits_processor=LogitsProcessorList([make_presence(1.5, n_prompt)]),
                             eos_token_id=stop, pad_token_id=pad)
        for i in range(len(chunk)):
            new = out[i, n_prompt:].tolist()
            ended = False
            for j, t in enumerate(new):
                if t in stop:
                    new, ended = new[:j], True
                    break
            prompt_ids = enc.input_ids[i][enc.attention_mask[i].bool()].tolist()
            outs.append({"id": chunk[i]["id"], "prompt_ids": prompt_ids, "new_ids": new, "ended": ended})
    return outs


# ─── 指标 ────────────────────────────────────────────────────────────────
def repeated_mask(ids, n=NGRAM):
    """实现选择 a)：位置 t 为"重复" = 以 t 结尾的 n 个 token 在更早位置出现过。"""
    seen, mask = set(), [False] * len(ids)
    for t in range(n - 1, len(ids)):
        g = tuple(ids[t - n + 1:t + 1])
        if g in seen:
            mask[t] = True
        else:
            seen.add(g)
    return mask


def loop_onset(ids, max_period=400, min_len=96):
    """第 2 版的循环判定（2026-11-15 自检后改）：回答的结尾是否在周期性重复。
    找最小周期 p，使结尾至少 max(4p, 160) 个 token 里 ≥90% 满足 ids[t] == ids[t-p]；
    再向前按 p 一块一块回溯，块内匹配率 ≥80% 就算仍在循环里。返回 (循环起点, 周期) 或 None。
    第 1 版的 8-gram 标记把正常推理里对题目和公式的复述也算成了"重复"，所以弃用于判定循环。"""
    a = np.asarray(ids)
    n = len(a)
    for p in range(1, min(max_period, n // 4) + 1):
        w = max(4 * p, 160)
        if n < w + p:
            break
        m = a[p:] == a[:-p]                       # m[i] 对应位置 i+p
        if m[-w:].mean() < 0.9:
            continue
        start = n - w
        while start - p >= p and m[start - p - p:start - p].mean() >= 0.8:
            start -= p
        if n - start >= min_len:
            return start, p
    return None


def loop_masks(ids, margin=64):
    """排序用的两类 token：循环内（起点之后一个周期起）/ 正常（循环起点前 margin 个 token 之前，或没有循环的整条回答）。"""
    lo = loop_onset(ids)
    rep = [False] * len(ids)
    new = [True] * len(ids)
    if lo:
        start, p = lo
        for t in range(len(ids)):
            rep[t] = t >= start + p
            new[t] = t < start - margin
    return rep, new


def rep_rate(text):
    """与 tgi_app.repetition_RR 相同的定义：1 − 不同词数 / 总词数（按空格切、小写）。"""
    toks = [t for t in text.lower().split() if t]
    return 1.0 - len(set(toks)) / len(toks) if toks else 0.0


def metrics(tok, outs, cap):
    rows = []
    for o in outs:
        text = tok.decode(o["new_ids"], skip_special_tokens=False)
        m = repeated_mask(o["new_ids"])
        lo = None if o["ended"] else loop_onset(o["new_ids"])
        rows.append({"n": len(o["new_ids"]), "hit_cap": (not o["ended"]) and len(o["new_ids"]) >= cap,
                     "hard_loop": lo is not None, "onset": lo[0] if lo else None,
                     "in_loop": (len(o["new_ids"]) - lo[0]) / len(o["new_ids"]) if lo else 0.0,
                     "loop_share": sum(m) / max(1, len(m)), "rep_rate": rep_rate(text),
                     "closed_think": "</think>" in text, "boxed": "\\boxed" in text.split("</think>")[-1]
                     if "</think>" in text else False})
    n = len(rows)
    onsets = [r["onset"] for r in rows if r["onset"] is not None]
    return {"n": n, "hit_cap": sum(r["hit_cap"] for r in rows) / n,
            "hard_loop": sum(r["hard_loop"] for r in rows) / n,
            "in_loop": float(np.mean([r["in_loop"] for r in rows])),
            "onset_median": float(np.median(onsets)) if onsets else None,
            "onsets": [r["onset"] for r in rows],          # 每条回答的循环起点（未循环为 None），画曲线用
            "lengths": [r["n"] for r in rows],
            "loop_share": float(np.mean([r["loop_share"] for r in rows])),
            "loop_gt_half": sum(r["loop_share"] > 0.5 for r in rows) / n,
            "rep_rate": float(np.mean([r["rep_rate"] for r in rows])),
            "closed_think": sum(r["closed_think"] for r in rows) / n,
            "boxed": sum(r["boxed"] for r in rows) / n,
            "mean_tokens": float(np.mean([r["n"] for r in rows]))}


def fmt(m):
    on = f"{m['onset_median']:.0f}" if m.get("onset_median") is not None else "-"
    return (f"陷入循环的回答 {m['hard_loop']:.0%} · 循环内 token 占比 {m['in_loop']:.1%} · 循环起点中位 {on} · "
            f"撑满上限 {m['hit_cap']:.0%} · RepRate {m['rep_rate']:.3f} · 思考收尾 {m['closed_think']:.0%} · "
            f"给出 boxed {m['boxed']:.0%} · 平均 {m['mean_tokens']:.0f} token")


# ─── MIT 的标准答案 ──────────────────────────────────────────────────────
def mit_columns(n_inter=3584):
    """从公开的 edit.npz 取出每层被改的列（= MLP 神经元编号）。"""
    z = np.load(REF / "edit.npz")
    man = json.loads((REF / "manifest.json").read_text())
    cols = {}
    for i, t in enumerate(man["tensors"]):
        layer = int(t["name"].split(".layers.")[1].split(".")[0])
        cols[layer] = sorted(set((z[f"t{i}_idx"] % n_inter).tolist()))
        assert len(cols[layer]) == t["changed_columns"], (layer, len(cols[layer]))
    return cols


def mit_edited_weights(model):
    """按公开的差分重建 MIT 改后的 down_proj（bfloat16 位级一致，校验 sha256）。返回 {layer: tensor}。"""
    z = np.load(REF / "edit.npz")
    man = json.loads((REF / "manifest.json").read_text())
    out = {}
    for i, t in enumerate(man["tensors"]):
        layer = int(t["name"].split(".layers.")[1].split(".")[0])
        w = model.model.layers[layer].mlp.down_proj.weight.detach().cpu()
        bits = w.view(torch.int16).numpy().copy()
        assert hashlib.sha256(bits.tobytes()).hexdigest() == t["base_sha256"], "本机模型不是 MIT 用的基准权重"
        flat = bits.reshape(-1).view(np.uint16)
        flat[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        assert hashlib.sha256(bits.tobytes()).hexdigest() == t["edited_sha256"], "重建结果与 MIT 评测权重不一致"
        out[layer] = torch.from_numpy(bits).view(torch.bfloat16).reshape(w.shape)
    return out


# ─── 阶段 collect ────────────────────────────────────────────────────────
@torch.no_grad()
def collect_activations(model, outs, pad_id, batch=2):
    """不采样的前向：累加每层 MLP 神经元在 重复 / 新 token 上的激活（和、平方和），以及残差流均值。"""
    layers = model.model.layers
    L, I, D = len(layers), layers[0].mlp.down_proj.in_features, layers[0].mlp.down_proj.out_features
    acc = {c: {"s": torch.zeros(L, I, device="cuda", dtype=torch.float64),
               "q": torch.zeros(L, I, device="cuda", dtype=torch.float64),
               "h": torch.zeros(L, D, device="cuda", dtype=torch.float64), "n": 0} for c in ("rep", "new")}
    cur = {}

    def mlp_hook(li):
        def fn(_m, args):
            x = args[0].float()                                   # [B, T, I]
            for c in ("rep", "new"):
                m = cur[c].unsqueeze(-1)
                acc[c]["s"][li] += (x * m).sum((0, 1)).double()
                acc[c]["q"][li] += (x * x * m).sum((0, 1)).double()
        return fn

    def out_hook(li):
        def fn(_m, _args, output):
            h = (output[0] if isinstance(output, tuple) else output).float()   # 该层输出的残差流 [B, T, D]
            for c in ("rep", "new"):
                acc[c]["h"][li] += (h * cur[c].unsqueeze(-1)).sum((0, 1)).double()
        return fn

    handles = [l.mlp.down_proj.register_forward_pre_hook(mlp_hook(i)) for i, l in enumerate(layers)]
    handles += [l.register_forward_hook(out_hook(i)) for i, l in enumerate(layers)]
    order = sorted(range(len(outs)), key=lambda i: len(outs[i]["prompt_ids"]) + len(outs[i]["new_ids"]))
    for k in range(0, len(order), batch):
        chunk = [outs[i] for i in order[k:k + batch]]
        T = max(len(o["prompt_ids"]) + len(o["new_ids"]) for o in chunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        rep = torch.zeros((len(chunk), T))
        new = torch.zeros((len(chunk), T))
        for j, o in enumerate(chunk):
            seq = o["prompt_ids"] + o["new_ids"]
            ids[j, :len(seq)] = torch.tensor(seq)
            att[j, :len(seq)] = 1
            mr, mn = loop_masks(o["new_ids"]) if not o["ended"] else ([False] * len(o["new_ids"]),
                                                                      [True] * len(o["new_ids"]))
            p = len(o["prompt_ids"])
            rep[j, p:p + len(mr)] = torch.tensor(mr, dtype=torch.float32)
            new[j, p:p + len(mn)] = torch.tensor(mn, dtype=torch.float32)
        cur["rep"], cur["new"] = rep.cuda(), new.cuda()
        acc["rep"]["n"] += int(rep.sum())
        acc["new"]["n"] += int(new.sum())
        # 只跑到最后一层隐状态，不算词表 logits（4 条 × 8000 token × 24.8 万词的 logits 要 16GB，
        # 2026-11-15 第二版第一次运行就是在这里撞上本进程的内存上限）
        model.model(input_ids=ids.cuda(), attention_mask=att.cuda())
    for h in handles:
        h.remove()
    res = {}
    for c in ("rep", "new"):
        n = max(1, acc[c]["n"])
        mean = acc[c]["s"] / n
        res[c] = {"mean": mean.cpu().numpy(), "var": (acc[c]["q"] / n - mean ** 2).clamp_min(0).cpu().numpy(),
                  "h": (acc[c]["h"] / n).cpu().numpy(), "n": acc[c]["n"]}
    return res


def phase_collect(args, model, tok, problems):
    out_dir = Path(args.out)
    t0 = time.time()
    ids_file = out_dir / "collect_ids.pt"
    if ids_file.exists():               # 生成很慢（8192 上限下约半小时），已有结果就直接用
        outs = torch.load(ids_file)
        log(f"沿用已保存的生成结果 {ids_file.name}（{len(outs)} 条）")
    else:
        outs = generate(model, tok, problems[:args.n], args.cap, args.batch, args.seed)
        torch.save([{k: o[k] for k in ("id", "prompt_ids", "new_ids", "ended")} for o in outs], ids_file)
    m = metrics(tok, outs, args.cap)
    log(f"[未修改模型，{args.n} 题，上限 {args.cap}] {fmt(m)}  ({time.time() - t0:.0f}s)")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    act = collect_activations(model, outs, pad)
    log(f"激活统计：循环内 token {act['rep']['n']:,} 个，正常 token {act['new']['n']:,} 个")
    np.savez_compressed(out_dir / "collect.npz", rep_mean=act["rep"]["mean"], rep_var=act["rep"]["var"],
                        new_mean=act["new"]["mean"], new_var=act["new"]["var"],
                        rep_h=act["rep"]["h"], new_h=act["new"]["h"],
                        n=np.array([act["rep"]["n"], act["new"]["n"]]))
    (out_dir / "collect.json").write_text(json.dumps(
        {"metrics": m, "cap": args.cap, "n": args.n, "seed": args.seed,
         "samples": [{"id": o["id"], "n": len(o["new_ids"]), "ended": o["ended"],
                      "loop": None if o["ended"] else loop_onset(o["new_ids"]),
                      "tail": tok.decode(o["new_ids"][-120:])} for o in outs]},
        ensure_ascii=False, indent=1), "utf-8")


# ─── 阶段 rank ───────────────────────────────────────────────────────────
def hypergeom_p(k, K, n, N):
    """从 N 个里随机取 n 个，命中 K 个目标中至少 k 个的概率。"""
    from math import comb
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / comb(N, n)


def rank_scores(out_dir, model):
    z = np.load(out_dir / "collect.npz")
    d = (z["rep_mean"] - z["new_mean"]) / np.sqrt((z["rep_var"] + z["new_var"]) / 2 + 1e-12)     # 实现选择 b)
    # 事后探索：每个神经元"写入幽灵方向"的量 = 重复态平均激活 × (该神经元的输出列 · 幽灵方向)
    # 幽灵方向按工程框架 §4.2：v = E[h_ghost − h_normal]，取该层输出的残差流
    writer = np.zeros_like(d)
    for li, layer in enumerate(model.model.layers):
        v = z["rep_h"][li] - z["new_h"][li]
        v = v / (np.linalg.norm(v) + 1e-12)
        W = layer.mlp.down_proj.weight.detach().float().cpu().numpy()        # [D, I]
        writer[li] = z["rep_mean"][li] * (v @ W)
    return d, writer


def phase_rank(args, model):
    out_dir = Path(args.out)
    mit = mit_columns()
    d, writer = rank_scores(out_dir, model)
    L, I = d.shape
    rep = {"mit_columns": {str(k): v for k, v in mit.items()}}
    log("== 标准答案：MIT 改动的神经元 ==")
    for l in EDIT_LAYERS:
        log(f"   第 {l} 层 {len(mit[l])} 个")

    # P1：按领域的候选集
    if args.probe_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import tgi_probe as P
        pj = json.loads((Path(args.probe_dir) / "probe.json").read_text("utf-8"))
        neur = np.load(Path(args.probe_dir) / "probe_raw.npz")["neuron"].astype(np.float32)
        neur = neur.reshape(neur.shape[0], -1)
        labels = np.array([r["domain"] for r in pj["records"]])
        mit_flat = {l * I + c for l in EDIT_LAYERS for c in mit[l]}
        log("== P1 按领域的候选集（判定：数学 Top-1000 命中 ≥ 8 成立，≤ 3 不成立；随机期望 ≈ 1.5） ==")
        rep["P1"] = {}
        for dom in P.DOMAINS:
            s = P.selectivity(neur, labels, dom)
            order = np.argsort(s)[::-1]
            hits = {k: len(set(order[:k].tolist()) & mit_flat) for k in (200, 1000, 5000)}
            pctl = float(np.mean([(s < s[i]).mean() for i in mit_flat]))
            rep["P1"][dom] = {"hits": hits, "mean_percentile": pctl}
            log(f"   {P.DOMAIN_ZH[dom]}：Top-200 命中 {hits[200]}，Top-1000 命中 {hits[1000]}，Top-5000 命中 {hits[5000]}；"
                f"MIT 神经元在该领域得分里的平均百分位 {pctl:.0%}")
    else:
        log("（没有给 --probe_dir，跳过 P1）")

    # P2：重复 − 新 的激活差
    log("== P2 重复态 vs 正常态激活差（判定：两层各取前 64，合计命中 ≥ 16 成立，≤ 5 不成立；随机期望 ≈ 1.1/层） ==")
    rep["P2"], total = {}, 0
    for l in EDIT_LAYERS:
        order = np.argsort(d[l])[::-1]
        top = set(order[:64].tolist())
        hit = len(top & set(mit[l]))
        total += hit
        pctl = float(np.mean([(d[l] < d[l][c]).mean() for c in mit[l]]))
        hit_abs = len(set(np.argsort(np.abs(d[l]))[::-1][:64].tolist()) & set(mit[l]))
        hit_neg = len(set(np.argsort(d[l])[:64].tolist()) & set(mit[l]))
        rep["P2"][str(l)] = {"hits_top64": hit, "p": hypergeom_p(hit, 64, 64, I), "mean_percentile": pctl,
                             "hits_abs": hit_abs, "hits_negative": hit_neg, "top64": sorted(top)}
        log(f"   第 {l} 层：前 64 命中 {hit}/64（随机得到这么多或更多的概率 {hypergeom_p(hit, 64, 64, I):.2g}）；"
            f"MIT 神经元的平均百分位 {pctl:.0%}；按绝对值取前 64 命中 {hit_abs}，取最负的 64 命中 {hit_neg}")
    rep["P2"]["total"] = total
    log(f"   合计 {total}/128")
    flat = np.argsort(d.reshape(-1))[::-1][:128]
    hist = np.bincount(flat // I, minlength=L)
    rep["P2"]["global_top128_layer_hist"] = hist.tolist()
    log("   不限定层、全模型取前 128：各层个数 " + " ".join(f"L{i}:{n}" for i, n in enumerate(hist) if n))

    # 事后探索：写入幽灵方向的量
    log("== 附加（事后探索，未预注册）：按「写入幽灵方向的量」排序 ==")
    rep["writer"], total_w = {}, 0
    for l in EDIT_LAYERS:
        order = np.argsort(writer[l])[::-1]
        top = set(order[:64].tolist())
        hit = len(top & set(mit[l]))
        total_w += hit
        hit_abs = len(set(np.argsort(np.abs(writer[l]))[::-1][:64].tolist()) & set(mit[l]))
        rep["writer"][str(l)] = {"hits_top64": hit, "p": hypergeom_p(hit, 64, 64, I), "hits_abs": hit_abs,
                                 "top64": sorted(top)}
        log(f"   第 {l} 层：前 64 命中 {hit}/64（概率 {hypergeom_p(hit, 64, 64, I):.2g}）；按绝对值取前 64 命中 {hit_abs}")
    layer_mass = np.sort(writer, axis=1)[:, ::-1][:, :64].sum(1)
    rep["writer"]["layer_mass_top64"] = layer_mass.tolist()
    top_layers = np.argsort(layer_mass)[::-1][:6]
    log("   每层前 64 个神经元的写入量之和，最大的 6 层：" + " ".join(f"L{i}({layer_mass[i]:.2f})" for i in top_layers))
    rep["writer"]["total"] = total_w
    (out_dir / "rank.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), "utf-8")
    np.savez_compressed(out_dir / "rank_scores.npz", d=d, writer=writer)


# ─── 阶段 ablate ─────────────────────────────────────────────────────────
class ZeroNeurons:
    """论文 §1.3 的 θ\\n_i：把选中的 MLP 神经元置零（置零它们进入 down_proj 的激活）。"""

    def __init__(self, model, cols):
        self.handles = []
        for layer, c in cols.items():
            idx = torch.tensor(sorted(c), device="cuda")
            self.handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(self._fn(idx)))

    @staticmethod
    def _fn(idx):
        def fn(_m, args):
            x = args[0].clone()
            x[..., idx] = 0
            return (x,)
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def phase_ablate(args, model, tok, problems):
    out_dir = Path(args.out)
    I = model.model.layers[0].mlp.down_proj.in_features
    held = problems[args.n:args.n + args.n_eval]              # 与排序用的题目不重叠
    mit = mit_columns()
    rng = np.random.default_rng(args.seed)
    sets = {
        "base": ("不改动", None),
        "rand": ("置零：同层随机 64", {l: rng.choice(I, 64, replace=False).tolist() for l in EDIT_LAYERS}),
        "mit_zero": ("置零：MIT 的 128 个神经元", {l: mit[l] for l in EDIT_LAYERS}),
    }
    if (out_dir / "rank.json").exists():      # 自己选出的神经元组要等排序阶段完成才有
        rank = json.loads((out_dir / "rank.json").read_text("utf-8"))
        sets["act"] = ("置零：TGI 激活差前 64（P2）", {l: rank["P2"][str(l)]["top64"] for l in EDIT_LAYERS})
        sets["writer"] = ("置零：写入量前 64", {l: rank["writer"][str(l)]["top64"] for l in EDIT_LAYERS})
    # 盲测：不使用 MIT 的层号和神经元，只用自有的写入量排序选 2 层 × 64 个
    blind_dirs = {}
    if (out_dir / "rank_scores.npz").exists():
        writer = np.load(out_dir / "rank_scores.npz")["writer"]                  # [L, I]
        mass = np.sort(writer, axis=1)[:, ::-1][:, :64].sum(1)
        blind_layers = [int(l) for l in np.argsort(mass)[::-1][:2]]
        blind_cols = {l: np.argsort(writer[l])[::-1][:64].tolist() for l in blind_layers}
        rng_b = np.random.default_rng(args.seed + 7)
        sets["blind"] = (f"盲测：置零自选层 {blind_layers} 的写入量前 64", blind_cols)
        sets["blind_rand"] = (f"盲测对照：层 {blind_layers} 随机置零各 64",
                              {l: rng_b.choice(I, 64, replace=False).tolist() for l in blind_layers})
        z = np.load(out_dir / "collect.npz")
        for l in blind_layers:
            v = z["rep_h"][l] - z["new_h"][l]
            blind_dirs[l] = v / (np.linalg.norm(v) + 1e-12)
    res_file = out_dir / "ablate.json"
    results = json.loads(res_file.read_text("utf-8")) if res_file.exists() else {}
    for key in [c for c in args.conds.split(",") if c]:
        t0 = time.time()
        if key == "blind_proj":          # 工程框架 §4.3：从选中神经元的输出列里减去幽灵方向上的分量
            name = f"盲测：自选层 {blind_layers} 的写入量前 64 减去幽灵方向"
            saved = {}
            for l in blind_layers:
                p = model.model.layers[l].mlp.down_proj.weight
                saved[l] = p.data.clone()
                W = p.data.float()
                v = torch.tensor(blind_dirs[l], dtype=torch.float32, device=W.device)
                cols = torch.tensor(blind_cols[l], device=W.device)
                sub = W[:, cols]
                W[:, cols] = sub - torch.outer(v, v @ sub)
                p.data.copy_(W.to(p.dtype))
            outs = generate(model, tok, held, args.cap, args.batch, args.seed + 1000)
            for l, w in saved.items():
                model.model.layers[l].mlp.down_proj.weight.data.copy_(w)
        elif key == "mit_edit":          # 参照：MIT 的实际改动（从输出列里减去方向，不是置零）
            name = "参照：MIT 的实际改动"
            edited = mit_edited_weights(model)
            saved = {}
            for l, w in edited.items():
                p = model.model.layers[l].mlp.down_proj.weight
                saved[l] = p.data.clone()
                p.data.copy_(w.to(p.device))
            outs = generate(model, tok, held, args.cap, args.batch, args.seed + 1000)
            for l, w in saved.items():
                model.model.layers[l].mlp.down_proj.weight.data.copy_(w)
        else:
            name, cols = sets[key]
            z = ZeroNeurons(model, cols) if cols else None
            outs = generate(model, tok, held, args.cap, args.batch, args.seed + 1000)
            if z:
                z.remove()
        results[name] = metrics(tok, outs, args.cap)
        log(f"[{name}] {fmt(results[name])}  ({time.time() - t0:.0f}s)")
        res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")


@torch.no_grad()
def phase_selftest(model, tok):
    """自检置零钩子：置零神经元 j 的激活，应当与把 down_proj 的第 j 列清零完全等价，并且确实改变输出。"""
    mit = mit_columns()
    ids = tok("Wait, 0.666 = 666/999. Wait, 0.666 = 666/999. Wait,", return_tensors="pt").input_ids.cuda()
    base = model(ids).logits.float()
    z = ZeroNeurons(model, {l: mit[l] for l in EDIT_LAYERS})
    hooked = model(ids).logits.float()
    z.remove()
    after = model(ids).logits.float()
    saved = {}
    for l in EDIT_LAYERS:
        p = model.model.layers[l].mlp.down_proj.weight
        saved[l] = p.data.clone()
        p.data[:, mit[l]] = 0
    colzero = model(ids).logits.float()
    for l, w in saved.items():
        model.model.layers[l].mlp.down_proj.weight.data.copy_(w)
    log(f"钩子置零 与 不改动 的最大差：{(hooked - base).abs().max():.4f}（应明显大于 0）")
    log(f"钩子置零 与 权重列清零 的最大差：{(hooked - colzero).abs().max():.6f}（应约等于 0）")
    log(f"移除钩子后 与 不改动 的最大差：{(after - base).abs().max():.6f}（应等于 0）")
    edited = mit_edited_weights(model)
    for l, w in edited.items():
        p = model.model.layers[l].mlp.down_proj.weight
        saved[l] = p.data.clone()
        p.data.copy_(w.to(p.device))
    ed = model(ids).logits.float()
    for l, w in saved.items():
        model.model.layers[l].mlp.down_proj.weight.data.copy_(w)
    log(f"MIT 实际改动 与 不改动 的最大差：{(ed - base).abs().max():.4f}（哈希已校验；应明显大于 0）")
    log(f"恢复权重后 与 不改动 的最大差：{(model(ids).logits.float() - base).abs().max():.6f}（应等于 0）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["collect", "rank", "ablate", "selftest"], required=True)
    ap.add_argument("--model", default=os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B"))
    ap.add_argument("--n", type=int, default=48, help="collect 用的题数（取练习题的前 n 道）")
    ap.add_argument("--n_eval", type=int, default=32, help="ablate 用的题数（紧接其后的 n_eval 道）")
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_dir", default="")
    ap.add_argument("--conds", default="base,mit_edit,rand,mit_zero,writer,act",
                    help="ablate 的条件，逗号分隔：base mit_edit rand mit_zero writer act")
    ap.add_argument("--max_mem_gb", type=float, default=24)
    ap.add_argument("--out", default=str(ROOT / "data" / "rcs_mit_Qwen3.5-0.8B_v2"))
    args = ap.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_mem_gb / total))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model)
    problems = [json.loads(l) for l in open(REF / "math_train.jsonl", encoding="utf-8")]
    if args.phase == "collect":
        phase_collect(args, model, tok, problems)
    elif args.phase == "rank":
        phase_rank(args, model)
    elif args.phase == "selftest":
        phase_selftest(model, tok)
    else:
        phase_ablate(args, model, tok, problems)
    log(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB · 输出 {args.out}")


if __name__ == "__main__":
    main()
