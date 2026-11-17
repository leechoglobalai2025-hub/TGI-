#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 权重测试 A / B（2026-11-16）— 只读权重，不生成、不改模型、不用 GPU。
判定标准预先写在 SESSION_20261116_TGI_Weights.md。

  A  第 23 层盲测选出的 64 个神经元是否为置信度调节型（Stolfo 等 2024 的两个特征：
     输出列范数大；输出列的能量大部分落在词表投影的近零空间）。
  B  Base → 后训练的权重差 ΔW：逐层改动量（B1）、与幽灵方向的对齐（B2）、
     神经元级与 MIT 128 个神经元的重合（B3）。

实现时的选择（论文没有规定）：
  a) 有效词表投影 = 输出嵌入（与输入嵌入共享）× 最后归一化层的增益；
  b) "近零空间" = 该矩阵奇异值最小的 5% 方向（1024 维里的 51 个）；
  c) 幽灵方向沿用昨天的定义：v_l = 循环态残差均值 − 正常态残差均值（第 l 层输出处）。

用法：python tgi_weights.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tgi_rcs import hypergeom_p, mit_columns  # noqa: E402

ROOT = Path(os.environ.get("TGI_RCS_ROOT", str(Path(__file__).resolve().parent.parent)))
POST = os.path.join(os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B"), "model.safetensors-00001-of-00001.safetensors")
BASE = os.path.join(os.environ.get("TGI_BASE_DIR", "models/Qwen3.5-0.8B-Base"), "model.safetensors-00001-of-00001.safetensors")
RCS = ROOT / "data" / "rcs_mit_Qwen3.5-0.8B_v2"
OUT = ROOT / "data" / "weights_20261116"
LM = "model.language_model."
NULL_FRAC = 0.05
TOP_FRAC = 0.05


def log(msg=""):
    print(msg, flush=True)


def down(sd, l):
    return sd[f"{LM}layers.{l}.mlp.down_proj.weight"].double()          # [D, I]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    post, base = load_file(POST), load_file(BASE)
    L = 1 + max(int(k.split(".layers.")[1].split(".")[0]) for k in post if ".layers." in k and k.startswith(LM))
    D, I = down(post, 0).shape
    writer = np.load(RCS / "rank_scores.npz")["writer"]                 # [L, I]
    z = np.load(RCS / "collect.npz")
    ghost = z["rep_h"] - z["new_h"]                                     # [L, D]
    ghost = ghost / (np.linalg.norm(ghost, axis=1, keepdims=True) + 1e-12)
    mit = mit_columns(I)
    res = {"L": L, "D": D, "I": I}
    log(f"结构：{L} 层，残差维度 {D}，每层 {I} 个 MLP 神经元")

    # ═══ A ═══════════════════════════════════════════════════════════════
    log("\n══ A：第 23 层自选神经元是否为置信度调节型 ══")
    gain = post[f"{LM}norm.weight"].double()
    log(f"最后归一化层增益：均值 {gain.mean():.3f}，最小 {gain.min():.3f}，最大 {gain.max():.3f}")
    E = post[f"{LM}embed_tokens.weight"].double() * gain                # 有效词表投影 [V, D]
    has_head = [k for k in post if "lm_head" in k]
    log(f"单独的输出头张量：{has_head if has_head else '无（与输入嵌入共享）'}")
    evals, evecs = torch.linalg.eigh(E.T @ E)                           # 升序
    sv = evals.clamp_min(0).sqrt()
    k = int(round(D * NULL_FRAC))
    V0 = evecs[:, :k]                                                   # 奇异值最小的 k 个方向
    log(f"奇异值：最小 {sv[0]:.2f}，第 {k} 小 {sv[k - 1]:.2f}，中位 {sv[D // 2]:.2f}，最大 {sv[-1]:.2f}")
    res["A"] = {"k": k, "sv_min": float(sv[0]), "sv_k": float(sv[k - 1]), "sv_median": float(sv[D // 2]),
                "sv_max": float(sv[-1]), "sv_smallest20": sv[:20].tolist(), "groups": {}}
    n_top = int(round(I * TOP_FRAC))

    def group_stats(l, cols, name):
        W = down(post, l)
        norm = W.norm(dim=0)
        rho = (V0.T @ W).pow(2).sum(0) / norm.pow(2)
        rho_rank = torch.argsort(rho, descending=True)
        top = set(rho_rank[:n_top].tolist())
        hit = len(top & set(cols))
        c = torch.tensor(sorted(cols))
        norm_pct = float(np.median([(norm < norm[i]).double().mean().item() for i in cols]))
        rho_pct = float(np.median([(rho < rho[i]).double().mean().item() for i in cols]))
        both = len({i for i in cols if (norm < norm[i]).double().mean() >= 0.9 and (rho < rho[i]).double().mean() >= 0.9})
        r = {"layer": l, "n": len(cols), "hits_rho_top5pct": hit, "expected": len(cols) * TOP_FRAC,
             "p": hypergeom_p(hit, n_top, len(cols), I), "rho_median": float(rho[c].median()),
             "rho_layer_median": float(rho.median()), "rho_layer_max": float(rho.max()),
             "norm_pct_median": norm_pct, "rho_pct_median": rho_pct, "both_top10pct": both,
             "norm_median": float(norm[c].median()), "norm_layer_median": float(norm.median())}
        res["A"]["groups"][name] = r
        log(f"  {name}（{len(cols)} 个）：ρ 前 5% 命中 {hit}（随机期望 {r['expected']:.1f}，p={r['p']:.2g}）；"
            f"ρ 中位 {r['rho_median']:.3f}（本层中位 {r['rho_layer_median']:.3f}，本层最大 {r['rho_layer_max']:.3f}）；"
            f"ρ 百分位中位 {rho_pct:.0%}；范数百分位中位 {norm_pct:.0%}；范数与 ρ 同在前 10% 的 {both} 个")
        return r

    sel = {l: np.argsort(writer[l])[::-1][:64].tolist() for l in (23, 19, 15)}
    log(f"近零空间 = 奇异值最小的 {k} 个方向；随机方向的 ρ 期望 = {NULL_FRAC:.2f}")
    main_a = group_stats(23, sel[23], "第 23 层 自选 64（主判定）")
    rng = np.random.default_rng(42 + 7)                                 # 与昨天盲测随机对照相同的种子
    rand = {l: rng.choice(I, 64, replace=False).tolist() for l in (23, 19)}
    group_stats(23, rand[23], "第 23 层 昨天的随机对照 64")
    group_stats(19, sel[19], "第 19 层 自选 64")
    group_stats(19, mit[19], "第 19 层 MIT 64")
    group_stats(15, sel[15], "第 15 层 自选 64")
    group_stats(15, mit[15], "第 15 层 MIT 64")
    h = main_a["hits_rho_top5pct"]
    verdict_a = "成立" if h >= 13 else ("不成立" if h <= 6 else "不确定")
    res["A"]["verdict"] = verdict_a
    log(f"A 主判定：命中 {h}（≥13 成立，≤6 不成立）→ {verdict_a}")

    # ═══ B ═══════════════════════════════════════════════════════════════
    log("\n══ B：Base → 后训练的权重差 ΔW ══")
    same = sorted(post) == sorted(base) and all(post[k].shape == base[k].shape for k in post)
    lm_keys = [k for k in post if k.startswith(LM)]
    n_changed = sum(1 for k in lm_keys if not torch.equal(post[k], base[k]))
    log(f"B0：张量名和形状一致 = {same}；语言模型部分 {len(lm_keys)} 个张量里有 {n_changed} 个不同")
    res["B"] = {"B0_same_structure": same, "B0_changed": n_changed, "B0_total": len(lm_keys)}
    if not same or n_changed == 0:
        log("B0 不满足，B 不做。")
    else:
        for name in ("embed_tokens.weight", "norm.weight"):
            a, b = post[LM + name].double(), base[LM + name].double()
            log(f"  {name}：相对改动 {((a - b).norm() / b.norm()).item():.4f}")
        # B1：逐层相对改动量
        kinds = sorted({k.split(f".layers.0.")[1] for k in lm_keys if ".layers.0." in k and post[k].dim() == 2}
                       | {k.split(".layers.3.")[1] for k in lm_keys if ".layers.3." in k and post[k].dim() == 2})
        rel = {kd: [None] * L for kd in kinds}
        for l in range(L):
            for kd in kinds:
                key = f"{LM}layers.{l}.{kd}"
                if key in post:
                    a, b = post[key].double(), base[key].double()
                    rel[kd][l] = ((a - b).norm() / b.norm()).item()
        res["B"]["B1_rel"] = rel
        log("B1：每层相对改动量 ‖ΔW‖/‖W_Base‖（纯权重）")
        log("  层  " + "  ".join(f"{kd.replace('.weight', '').replace('self_attn.', 'attn.').replace('mlp.', ''):>12}"
                                 for kd in kinds))
        for l in range(L):
            log(f"  {l:>2}  " + "  ".join(f"{rel[kd][l]:>12.4f}" if rel[kd][l] is not None else f"{'-':>12}"
                                          for kd in kinds))
        dk = "mlp.down_proj.weight"
        order1 = sorted(range(L), key=lambda l: -rel[dk][l])
        res["B"]["B1_down_order"] = order1
        log(f"  down_proj 相对改动量从大到小的层：{order1}")
        log(f"  第 15 层名次 {order1.index(15) + 1}/{L}，第 19 层名次 {order1.index(19) + 1}/{L}，"
            f"第 23 层名次 {order1.index(23) + 1}/{L}")

        # B2：ΔW_down 与幽灵方向的对齐
        log("B2：e_l = ‖v_lᵀ ΔW_down‖² / ‖ΔW_down‖² × D（1 = 与随机方向无异）")
        e, absw, e_w = [], [], []
        for l in range(L):
            dW = down(post, l) - down(base, l)
            v = torch.tensor(ghost[l], dtype=torch.float64)
            proj = v @ dW                                               # [I]
            e.append((proj.pow(2).sum() / dW.pow(2).sum() * D).item())
            absw.append(proj.norm().item())
            Wp = down(post, l)
            e_w.append(((v @ Wp).pow(2).sum() / Wp.pow(2).sum() * D).item())   # 对照：后训练权重本身的对齐
        res["B"]["B2_e"], res["B"]["B2_abs"], res["B"]["B2_e_weight_itself"] = e, absw, e_w
        order2 = sorted(range(L - 1), key=lambda l: -e[l])              # 排除最后一层
        res["B"]["B2_order_excl_last"] = order2
        for l in range(L):
            log(f"  第 {l:>2} 层：e = {e[l]:7.2f}   ‖vᵀΔW‖ = {absw[l]:.4f}   （对照：W 本身的 e = {e_w[l]:6.2f}）"
                + ("   ← 最后一层，单独报告" if l == L - 1 else ""))
        r15, r19 = order2.index(15) + 1, order2.index(19) + 1
        verdict_b2 = "成立" if max(r15, r19) <= 4 else ("不成立" if min(r15, r19) > 8 else "不确定")
        res["B"]["B2_rank15"], res["B"]["B2_rank19"], res["B"]["B2_verdict"] = r15, r19, verdict_b2
        log(f"  第 0–{L - 2} 层按 e 从大到小：{order2}")
        log(f"B2 主判定：第 15 层第 {r15} 名，第 19 层第 {r19} 名（都在前 4 成立，都不在前 8 不成立）→ {verdict_b2}")
        order2b = sorted(range(L - 1), key=lambda l: -absw[l])
        res["B"]["B2_abs_order_excl_last"] = order2b
        log(f"  只报告：按绝对量 ‖vᵀΔW‖ 排序：{order2b}")

        # B3：神经元级
        log("B3：第 15、19 层按 down_proj 列相对改动量取前 64，与 MIT 128 个的重合（≥16 成立，≤5 不成立）")
        tot, tot_v, res["B"]["B3"] = 0, 0, {}
        for l in (15, 19):
            Wb = down(base, l)
            dW = down(post, l) - Wb
            relc = dW.norm(dim=0) / Wb.norm(dim=0)
            top = set(torch.argsort(relc, descending=True)[:64].tolist())
            hit = len(top & set(mit[l]))
            v = torch.tensor(ghost[l], dtype=torch.float64)
            pv = (v @ dW).abs()
            hit_v = len(set(torch.argsort(pv, descending=True)[:64].tolist()) & set(mit[l]))
            hit_sel = len(top & set(sel[l]))
            pct = float(np.mean([(relc < relc[i]).double().mean().item() for i in mit[l]]))
            tot += hit
            tot_v += hit_v
            res["B"]["B3"][str(l)] = {"hits_rel": hit, "hits_ghostproj": hit_v, "hits_rel_vs_selfsel": hit_sel,
                                      "mit_mean_pct": pct, "relc_median": float(relc.median())}
            log(f"  第 {l} 层：相对改动量前 64 命中 MIT {hit}（p={hypergeom_p(hit, 64, 64, I):.2g}），命中昨天自选 {hit_sel}；"
                f"MIT 神经元的平均百分位 {pct:.0%}；变体 |v·Δw| 前 64 命中 MIT {hit_v}")
        verdict_b3 = "成立" if tot >= 16 else ("不成立" if tot <= 5 else "不确定")
        res["B"]["B3_total"], res["B"]["B3_total_ghostproj"], res["B"]["B3_verdict"] = tot, tot_v, verdict_b3
        log(f"B3 主判定：合计 {tot}/128 → {verdict_b3}；变体合计 {tot_v}/128（只报告）")

    (OUT / "weights.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    log(f"\n输出 {OUT / 'weights.json'}")


if __name__ == "__main__":
    main()
