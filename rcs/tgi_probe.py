#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI Probe — 第一阶段：只测量模型内部量，不下判定。

用 transformers 直接加载本地模型（不经过 vLLM），逐步记录三样东西：
  1. 输出分布的精确熵 H_t = -Σ P(v|C_t) log2 P(v|C_t)      （工程框架 §5.1，全词表，不截断）
  2. 注意力 Gini 系数 G_t^(l,h)                              （工程框架 §5.2 的公式）
  3. MLP 神经元激活，按领域（代码/数学/文学/医学）汇总        （用户 2026-11-15 定：神经元 = MLP 神经元，
                                                              RCS 候选集按领域划分）

不计算、不输出：安全/危险判定、理想质量 M、盆地半径 r、逃逸衰减率 λ、幂等指数 ι、ICL 耦合度 E、RCS。
这些量的定义（阈值形式、"被捕获"判据、矩阵构造、基线模型）尚未由用户确定。

下面三处是实现时的选择，论文没有规定，需用户审定：
  a) "神经元激活" 取 MLP 里进入 down_proj 的值，即 act(gate_proj(x)) * up_proj(x) 的每一维。
  b) 领域选择性得分 = (该领域均值 − 其他领域均值) / sqrt((两边方差之和)/2)，只看正方向（更活跃）。
  c) 生成用采样（默认 temperature 0.5，固定种子），与 Scanner 的扫描设置一致。

用法：
    python tgi_probe.py --model models/Qwen3.5-0.8B-Base
输出目录：TGI Scanner/Export/probe_<时间>_<模型名>/  {probe.json, probe_raw.npz, report.md}
"""
import argparse
import json
import os
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

EXPORT_DIR = Path(os.environ.get("TGI_RCS_ROOT", str(Path(__file__).resolve().parent.parent))) / "data"
DOMAINS = ("code", "math", "literature", "medicine")
DOMAIN_ZH = {"code": "代码", "math": "数学", "literature": "文学", "medicine": "医学"}
TOPK = (50, 100, 200, 500, 1000)

# 每个领域 中/韩/英 各 3 条
PROMPTS = {
    "code": [
        ("zh", "用Python写一个二分查找函数，并说明时间复杂度。"),
        ("zh", "解释什么是死锁，给出一个简单的代码例子。"),
        ("zh", "写一个SQL查询，找出每个部门工资最高的员工。"),
        ("ko", "파이썬으로 리스트의 중복을 제거하는 함수를 작성하세요."),
        ("ko", "자바스크립트에서 Promise와 async/await의 차이를 설명하세요."),
        ("ko", "C 언어로 문자열을 뒤집는 함수를 작성하세요."),
        ("en", "Write a Python function that checks whether a string is a palindrome."),
        ("en", "Explain what a race condition is and show a short code example."),
        ("en", "Write a Bash script that counts the lines in every .txt file in a directory."),
    ],
    "math": [
        ("zh", "证明根号2是无理数。"),
        ("zh", "求函数 f(x)=x^3-3x 的极值点。"),
        ("zh", "解方程组：2x+y=7，x-y=2。"),
        ("ko", "1부터 100까지의 합을 구하는 방법을 설명하세요."),
        ("ko", "이차방정식 x^2-5x+6=0 을 풀어 보세요."),
        ("ko", "미분의 정의를 설명하고 x^2 의 도함수를 구하세요."),
        ("en", "Prove that the sum of two even numbers is even."),
        ("en", "Compute the integral of x*e^x dx."),
        ("en", "What is the probability of getting exactly two heads in three coin tosses?"),
    ],
    "literature": [
        ("zh", "写一首关于秋天的短诗。"),
        ("zh", "分析鲁迅《孔乙己》中孔乙己这个人物。"),
        ("zh", "用优美的语言描写一场夜雨。"),
        ("ko", "봄을 주제로 짧은 시를 써 주세요."),
        ("ko", "윤동주의 '서시'가 전하는 의미를 설명해 주세요."),
        ("ko", "바닷가의 새벽 풍경을 문학적으로 묘사해 주세요."),
        ("en", "Write a short poem about the sea at night."),
        ("en", "Discuss the theme of ambition in Shakespeare's Macbeth."),
        ("en", "Describe an old library in vivid literary prose."),
    ],
    "medicine": [
        ("zh", "高血压的常见症状和治疗方法有哪些？"),
        ("zh", "解释1型糖尿病和2型糖尿病的区别。"),
        ("zh", "感冒和流感有什么不同？"),
        ("ko", "고혈압의 원인과 치료 방법을 설명해 주세요."),
        ("ko", "빈혈의 증상과 원인은 무엇인가요?"),
        ("ko", "항생제 내성이 무엇인지 설명해 주세요."),
        ("en", "What are the symptoms and treatment of asthma?"),
        ("en", "Explain how vaccines train the immune system."),
        ("en", "What is the difference between a virus and a bacterium?"),
    ],
}


def log(msg):
    print(msg, flush=True)


# ─── 模型 ────────────────────────────────────────────────────────────────
def load_model(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(path)
    if any(a.startswith("Qwen3_5") and "ConditionalGeneration" in a for a in cfg.architectures or []):
        from transformers import Qwen3_5ForCausalLM as Cls  # 多模态权重只加载文本部分
    else:
        Cls = AutoModelForCausalLM
    try:  # optional local speed-up on some unified-memory hosts
        import gb10_fastload
        gb10_fastload.apply()
    except Exception:
        pass
    # eager：只有这种实现会返回注意力分布
    model = Cls.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def find_mlp_down_projs(model):
    """每层 MLP 的 down_proj（它的输入就是该层的 MLP 神经元激活）。按层号排序。"""
    out = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp.down_proj"):
            layer = int(name.split(".layers.")[1].split(".")[0])
            out.append((layer, mod))
    return [m for _, m in sorted(out, key=lambda x: x[0])]


class NeuronRecorder:
    """累加每层 MLP 神经元在所有 token 位置上的激活。"""

    def __init__(self, mods):
        self.sums = [None] * len(mods)
        self.count = 0
        self.handles = [m.register_forward_pre_hook(self._hook(i)) for i, m in enumerate(mods)]

    def _hook(self, i):
        def fn(_mod, args):
            x = args[0].detach().float()          # [1, T, I]
            s = x.sum(dim=(0, 1))
            self.sums[i] = s if self.sums[i] is None else self.sums[i] + s
            if i == 0:
                self.count += x.shape[1]
        return fn

    def reset(self):
        self.sums = [None] * len(self.sums)
        self.count = 0

    def mean(self):
        return torch.stack(self.sums).cpu().numpy() / max(1, self.count)   # [L, I]

    def remove(self):
        for h in self.handles:
            h.remove()


def gini_last_row(attn):
    """工程框架 §5.2：G = 1 − 2 Σ_i sort(Attn)_i · (t − i + 0.5) / (t · Σ_j Attn_j)，升序排序，i 从 1 起。
    attn: [1, heads, q, t]，取最后一个查询位置。返回 [heads]。"""
    x = attn[0, :, -1, :].float()
    t = x.shape[-1]
    xs, _ = torch.sort(x, dim=-1)
    w = t - torch.arange(1, t + 1, device=x.device, dtype=torch.float32) + 0.5
    return (1 - 2 * (xs * w).sum(-1) / (t * xs.sum(-1).clamp_min(1e-12))).cpu().numpy()


@torch.no_grad()
def run_prompt(model, tok, rec, prompt, max_new, temp, gen, stop):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    rec.reset()
    out = model(ids, use_cache=True, output_attentions=True)
    H, G, new = [], [], []
    for _ in range(max_new):
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, -1)
        H.append(float(-(logp.exp() * logp).sum() / math.log(2)))            # 精确熵（bit），与采样温度无关
        G.append(np.stack([gini_last_row(a) for a in out.attentions]))        # [注意力层数, heads]
        if temp > 0:
            nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1, generator=gen)
        else:
            nxt = logits.argmax().view(1)
        if int(nxt) in stop:
            break
        new.append(int(nxt))
        out = model(nxt.view(1, 1), past_key_values=out.past_key_values, use_cache=True,
                    output_attentions=True)
    return {"text": tok.decode(new, skip_special_tokens=True), "n_prompt": int(ids.shape[1]),
            "H": H, "G": np.stack(G).astype(np.float32), "neuron": rec.mean().astype(np.float32),
            "hit_max": len(new) >= max_new}


# ─── 汇总（只描述分布，不判定） ──────────────────────────────────────────
def pct(xs, q):
    return float(np.percentile(xs, q))


def selectivity(neur, labels, dom):
    """实现选择 b)：每个神经元对领域 dom 的选择性得分。neur: [N, L*I]。"""
    a = neur[labels == dom]
    b = neur[labels != dom]
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)


def summarize(results, n_layers, n_inter, attn_layers, seed):
    rep = {}
    # 1. 精确熵
    allH = np.concatenate([np.array(r["H"]) for r in results])
    mu, sd = float(allH.mean()), float(allH.std())
    rep["entropy"] = {
        "steps": int(allH.size), "mean": mu, "std": sd, "mean_minus_2std": mu - 2 * sd,
        "mean_minus_3std": mu - 3 * sd,
        "percentiles": {str(q): pct(allH, q) for q in (1, 5, 25, 50, 75, 95, 99)},
        "frac_below_0.01": float((allH < 0.01).mean()), "frac_below_0.1": float((allH < 0.1).mean()),
        "frac_above_1": float((allH > 1).mean()), "frac_above_3": float((allH > 3).mean()),
        "steps_below_mean_minus_2std": int((allH < mu - 2 * sd).sum()),
        "log2H": {"mean": float(np.log2(allH.clip(1e-6)).mean()), "std": float(np.log2(allH.clip(1e-6)).std())},
        "by_domain": {}, "by_lang": {},
    }
    for key, field in (("by_domain", "domain"), ("by_lang", "lang")):
        for v in sorted({r[field] for r in results}):
            h = np.concatenate([np.array(r["H"]) for r in results if r[field] == v])
            rep["entropy"][key][v] = {"mean": float(h.mean()), "median": pct(h, 50), "p95": pct(h, 95),
                                      "steps": int(h.size)}
    # 2. 注意力 Gini
    Gall = np.concatenate([r["G"] for r in results])                 # [总步数, A, heads]
    slopes, longest = [], []
    for r in results:
        g = r["G"]
        if len(g) < 8:
            continue
        x = np.arange(len(g)) - (len(g) - 1) / 2
        slopes.append((g * x[:, None, None]).sum(0) / (x ** 2).sum())   # 每步的 Gini 变化
        up = np.diff(g, axis=0) > 0
        run = np.zeros(g.shape[1:], dtype=int)
        best = np.zeros(g.shape[1:], dtype=int)
        for row in up:
            run = np.where(row, run + 1, 0)
            best = np.maximum(best, run)
        longest.append(best)
    rep["gini"] = {
        "attn_layers": attn_layers, "heads": int(Gall.shape[2]),
        "mean_by_layer_head": np.round(Gall.mean(0), 4).tolist(),
        "std_by_layer_head": np.round(Gall.std(0), 4).tolist(),
        "slope_per_step_by_layer_head": np.round(np.mean(slopes, 0), 6).tolist(),
        "longest_increasing_run_max": np.max(longest, 0).tolist(),
        "overall_mean": float(Gall.mean()), "overall_min": float(Gall.min()), "overall_max": float(Gall.max()),
    }
    # 3. 按领域的 MLP 神经元
    neur = np.stack([r["neuron"].reshape(-1) for r in results])      # [N, L*I]
    labels = np.array([r["domain"] for r in results])
    scores = {d: selectivity(neur, labels, d) for d in DOMAINS}
    # 对照：打乱领域标签后的得分（没有真实领域结构时的水平）
    rng = np.random.default_rng(seed)
    null_top = {k: [] for k in TOPK}
    null_max = []
    for _ in range(20):
        perm = rng.permutation(labels)
        s = selectivity(neur, perm, DOMAINS[0])
        srt = np.sort(s)[::-1]
        null_max.append(float(srt[0]))
        for k in TOPK:
            null_top[k].append(float(srt[k - 1]))
    third = n_layers / 3
    rep["neurons"] = {"total": int(neur.shape[1]), "layers": n_layers, "per_layer": n_inter,
                      "null_kth_score_mean": {str(k): float(np.mean(v)) for k, v in null_top.items()},
                      "null_max_mean": float(np.mean(null_max)), "domains": {}, "overlap": {}}
    tops = {}
    for d in DOMAINS:
        order = np.argsort(scores[d])[::-1]
        tops[d] = {k: set(order[:k].tolist()) for k in TOPK}
        info = {"kth_score": {str(k): float(scores[d][order[k - 1]]) for k in TOPK},
                "n_above_null_max": int((scores[d] > np.mean(null_max)).sum()), "by_k": {}}
        for k in TOPK:
            layers = order[:k] // n_inter
            hist = np.bincount(layers, minlength=n_layers)
            info["by_k"][str(k)] = {
                "early_mid_late": [int((layers < third).sum()), int(((layers >= third) & (layers < 2 * third)).sum()),
                                   int((layers >= 2 * third).sum())],
                "top_layer": int(hist.argmax()), "top_layer_share": float(hist.max() / k),
                "layer_hist": hist.tolist()}
        info["top10"] = [{"layer": int(i // n_inter), "neuron": int(i % n_inter),
                          "score": round(float(scores[d][i]), 2)} for i in order[:10]]
        rep["neurons"]["domains"][d] = info
    for k in TOPK:
        m = {}
        for i, a in enumerate(DOMAINS):
            for b in DOMAINS[i + 1:]:
                inter = len(tops[a][k] & tops[b][k])
                m[f"{a}|{b}"] = {"shared": inter, "jaccard": round(inter / (2 * k - inter), 4)}
        rep["neurons"]["overlap"][str(k)] = m
    return rep


def write_report(path, meta, rep, results):
    e, g, n = rep["entropy"], rep["gini"], rep["neurons"]
    L = [f"# TGI Probe 测量报告 — {meta['model_name']}", "",
         f"时间 {meta['time']} · {meta['n_prompts']} 条提示（4 领域 × 中韩英）· 每条最多 {meta['max_new']} token · "
         f"temperature {meta['temperature']} · 峰值显存 {meta['peak_mem_gib']} GiB", "",
         "只有测量结果，没有安全/危险判定。", "",
         "## 1. 输出分布的精确熵（§5.1）", "",
         f"- 共 {e['steps']} 步：均值 μ = {e['mean']:.3f} bit，标准差 σ = {e['std']:.3f}",
         f"- μ−2σ = {e['mean_minus_2std']:.3f}，μ−3σ = {e['mean_minus_3std']:.3f}；低于 μ−2σ 的步数：{e['steps_below_mean_minus_2std']}",
         "- 分位数：" + "，".join(f"p{q}={v:.3f}" for q, v in e["percentiles"].items()),
         f"- H<0.01 的步占 {e['frac_below_0.01']:.1%}，H<0.1 占 {e['frac_below_0.1']:.1%}，H>1 占 {e['frac_above_1']:.1%}，H>3 占 {e['frac_above_3']:.1%}",
         f"- log2(H)：均值 {e['log2H']['mean']:.2f}，标准差 {e['log2H']['std']:.2f}", "",
         "| 分组 | 步数 | 均值 | 中位数 | p95 |", "|---|---|---|---|---|"]
    for key in ("by_domain", "by_lang"):
        for k, v in e[key].items():
            L.append(f"| {DOMAIN_ZH.get(k, k)} | {v['steps']} | {v['mean']:.3f} | {v['median']:.3f} | {v['p95']:.3f} |")
    L += ["", "## 2. 注意力 Gini（§5.2）", "",
          f"- 有注意力分布的层：{g['attn_layers']}（其余层是线性注意力，没有注意力分布）",
          f"- 总体：均值 {g['overall_mean']:.3f}，范围 {g['overall_min']:.3f} ~ {g['overall_max']:.3f}", "",
          "| 层 | 各头 Gini 均值 | 各头每步斜率 ×1000 | 各头最长连续上升步数 |", "|---|---|---|---|"]
    for i, layer in enumerate(g["attn_layers"]):
        L.append(f"| {layer} | " + " ".join(f"{x:.2f}" for x in g["mean_by_layer_head"][i]) + " | "
                 + " ".join(f"{x * 1000:+.2f}" for x in g["slope_per_step_by_layer_head"][i]) + " | "
                 + " ".join(str(x) for x in g["longest_increasing_run_max"][i]) + " |")
    L += ["", "## 3. 按领域的 MLP 神经元", "",
          f"- 神经元总数 {n['total']}（{n['layers']} 层 × {n['per_layer']}）",
          f"- 对照（打乱领域标签 20 次）：最高得分平均 {n['null_max_mean']:.2f}；第 K 名得分平均 "
          + "，".join(f"K={k}: {v:.2f}" for k, v in n["null_kth_score_mean"].items()), "",
          "| 领域 | 得分高于对照最高分的神经元数 | 第 K 名得分（K=50/100/200/500/1000） | Top-200 在 前/中/后 三段层的分布 | Top-200 最集中的层 |",
          "|---|---|---|---|---|"]
    for d in DOMAINS:
        v = n["domains"][d]
        b = v["by_k"]["200"]
        L.append(f"| {DOMAIN_ZH[d]} | {v['n_above_null_max']} | " + " / ".join(f"{v['kth_score'][str(k)]:.2f}" for k in TOPK)
                 + f" | {b['early_mid_late'][0]} / {b['early_mid_late'][1]} / {b['early_mid_late'][2]} | "
                 f"第 {b['top_layer']} 层（{b['top_layer_share']:.0%}） |")
    L += ["", "领域候选集之间的重叠（共有神经元数）：", "", "| K | " + " | ".join(
        f"{DOMAIN_ZH[a]}∩{DOMAIN_ZH[b]}" for i, a in enumerate(DOMAINS) for b in DOMAINS[i + 1:]) + " |",
        "|---|" + "---|" * 6]
    for k in TOPK:
        L.append(f"| {k} | " + " | ".join(str(v["shared"]) for v in n["overlap"][str(k)].values()) + " |")
    L += ["", "## 4. 各条提示", "", "| 领域 | 语言 | 生成 token | 熵均值 | 熵中位数 | 回答开头 |", "|---|---|---|---|---|---|"]
    for r in results:
        h = np.array(r["H"])
        head = r["text"][:40].replace("\n", " ").replace("|", "/")
        L.append(f"| {DOMAIN_ZH[r['domain']]} | {r['lang']} | {len(h)}{'（到上限）' if r['hit_max'] else ''} | "
                 f"{h.mean():.3f} | {np.median(h):.3f} | {head} |")
    path.write_text("\n".join(L) + "\n", "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_mem_gb", type=float, default=16,
                    help="本进程 GPU 内存上限（统一内存下防止挤占系统内存）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("❌ CUDA 不可用")
        sys.exit(1)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_mem_gb / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    model_path = str(Path(args.model).expanduser())
    name = Path(model_path).name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else EXPORT_DIR / f"probe_{ts}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"加载 {name}（内存上限 {args.max_mem_gb:.0f} GiB）…")
    model, tok = load_model(model_path)
    mods = find_mlp_down_projs(model)
    if not mods:
        log("❌ 没找到 MLP 的 down_proj，这个模型结构暂不支持")
        sys.exit(1)
    n_layers, n_inter = len(mods), mods[0].in_features
    stop = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    rec = NeuronRecorder(mods)
    log(f"加载完成 {time.time() - t0:.0f}s · {n_layers} 层 × {n_inter} 个 MLP 神经元")

    # 哪些层有注意力分布（混合结构的模型只有部分层是 softmax 注意力）
    attn_layers = [i for i, layer in enumerate(model.model.layers) if hasattr(layer, "self_attn")]
    jobs = [(d, lang, p) for d in DOMAINS for lang, p in PROMPTS[d]]
    results = []
    for i, (d, lang, p) in enumerate(jobs, 1):
        t1 = time.time()
        r = run_prompt(model, tok, rec, p, args.max_new, args.temperature, gen, stop)
        r.update(domain=d, lang=lang, prompt=p)
        results.append(r)
        log(f"[{i}/{len(jobs)}] {DOMAIN_ZH[d]}/{lang} {len(r['H'])} 步 {time.time() - t1:.0f}s "
            f"熵均值 {np.mean(r['H']):.3f} · {r['text'][:30]!r}")
    rec.remove()
    n_attn = results[0]["G"].shape[1]
    if len(attn_layers) != n_attn:      # 层上没有 self_attn 属性时，只知道数量
        attn_layers = list(range(n_attn))

    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    meta = {"model": model_path, "model_name": name, "time": datetime.now().isoformat(timespec="seconds"),
            "n_prompts": len(jobs), "max_new": args.max_new, "temperature": args.temperature,
            "seed": args.seed, "peak_mem_gib": peak, "seconds": round(time.time() - t0)}
    rep = summarize(results, n_layers, n_inter, attn_layers, args.seed)
    np.savez_compressed(out_dir / "probe_raw.npz",
                        neuron=np.stack([r["neuron"] for r in results]).astype(np.float16),
                        **{f"gini_{i}": r["G"].astype(np.float16) for i, r in enumerate(results)})
    (out_dir / "probe.json").write_text(json.dumps({
        "meta": meta, "summary": rep,
        "records": [{k: r[k] for k in ("domain", "lang", "prompt", "text", "n_prompt", "H", "hit_max")}
                    for r in results]}, ensure_ascii=False), "utf-8")
    write_report(out_dir / "report.md", meta, rep, results)
    log(f"✅ 完成 {meta['seconds']}s · 峰值显存 {peak} GiB · 输出 {out_dir}")
    log("TGI_PROBE " + json.dumps({"out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
