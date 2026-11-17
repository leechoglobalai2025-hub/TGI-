#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 测试 C（2026-11-16）：分别只动第 15 层、只动第 19 层、只动第 23 层。
判定标准预先写在 SESSION_20261116_TGI_Weights.md。

神经元、方向、题目、随机种子全部沿用昨天（Export/rcs_mit_Qwen3.5-0.8B_v2/）：
  每层的 64 个神经元 = 该层写入量最大的 64 个（与昨天两次消融用的是同一组）；
  方向 = 该层输出处的单位幽灵方向；题目 = 练习题第 33–64 题；上限 8192。
所以结果可以直接和昨天的"不改动 17/32"等数字比较。

  z15 / z19 / z23：只把该层的 64 个神经元置零（论文 §1.3）。
  p15 / p19 / p23：只从该层的 64 个神经元的输出列里减去幽灵方向（工程框架 §4.3）。

用法：python tgi_layers.py --conds z23,z19,z15
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tgi_rcs as R  # noqa: E402
from tgi_blind2 import SubtractDirection  # noqa: E402

SRC = R.ROOT / "data" / "rcs_mit_Qwen3.5-0.8B_v2"
OUT = R.ROOT / "data" / "rcs_layers_20261116"
MODEL = os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B")
CAP, BATCH, SEED, K = 8192, 32, 42, 64
log = R.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="z23,z19,z15,p23,p19,p15")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, 24 / total))
    model, tok = R.load_model(MODEL)
    problems = [json.loads(x) for x in open(R.REF / "math_train.jsonl", encoding="utf-8")]
    held = problems[32:64]                                             # 与昨天的消融相同
    writer = np.load(SRC / "rank_scores.npz")["writer"]
    z = np.load(SRC / "collect.npz")
    res_file = OUT / "ablate.json"
    results = json.loads(res_file.read_text("utf-8")) if res_file.exists() else {}
    for key in [c for c in args.conds.split(",") if c]:
        kind, l = key[0], int(key[1:])
        cols = {l: [int(c) for c in np.argsort(writer[l])[::-1][:K]]}
        name = f"只动第 {l} 层：" + ("置零写入量前 64" if kind == "z" else "写入量前 64 减去幽灵方向")
        if name in results:
            log(f"[{name}] 已有结果，跳过")
            continue
        t0 = time.time()
        if kind == "z":
            h = R.ZeroNeurons(model, cols)
            undo = h.remove
        else:
            v = z["rep_h"][l] - z["new_h"][l]
            s = SubtractDirection(model, cols, {l: v / (np.linalg.norm(v) + 1e-12)})
            undo = s.restore
        outs = R.generate(model, tok, held, CAP, BATCH, SEED + 1000)
        undo()
        results[name] = R.metrics(tok, outs, CAP)
        torch.save([{k: o[k] for k in ("id", "new_ids", "ended")} for o in outs], OUT / f"gen_{key}.pt")
        log(f"[{name}] {R.fmt(results[name])}  ({time.time() - t0:.0f}s)")
        res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    log(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB · 输出 {OUT}")


if __name__ == "__main__":
    main()
