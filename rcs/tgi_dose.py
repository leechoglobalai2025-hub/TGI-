#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 生成确认（2026-11-16）：盲测选出的方向 + 神经元，剂量 α=6，在 D 的评测题（第 233–264 题）上生成。
规则与判定标准预先写在 SESSION_20261116_TGI_Weights.md。

  方向和神经元：今天 D 从第 201–232 题选出的（Export/rcs_blind2_20261116/），不使用 MIT 的层号和神经元编号。
  公式：W = W0 − α·outer(u, mask·(uᵀW0))，α=6（剂量取自 MIT 的最终改动，不是我们自己定出来的）。
  对照：同样的神经元 + 与我们的方向正交的随机单位方向，缩放到每层改动范数相同。
  已有参照（D，同样 32 题、同一随机种子）：不改动 19/32、MIT 的实际改动 1/32、置零 6/32、随机置零 11/32。

用法：python tgi_dose.py
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
from tgi_pairs import Edit, unit  # noqa: E402

B2 = R.ROOT / "data" / "rcs_blind2_20261116"
OUT = R.ROOT / "data" / "rcs_dose_20261116"
MODEL = os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B")
CAP, BATCH, SEED, ALPHA = 8192, 32, 42, 6.0
log = R.log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, 24 / total))
    model, tok = R.load_model(MODEL)
    problems = [json.loads(x) for x in open(R.REF / "math_train.jsonl", encoding="utf-8")]
    held = problems[232:264]
    sel = json.loads((B2 / "selection.json").read_text("utf-8"))
    layers = sorted(sel["layers"])
    assert layers == [15, 19], layers
    cols = {l: sel["cols"][str(l)] for l in layers}
    d2 = np.load(B2 / "dirs.npz")
    dirs = {l: unit(d2[f"L{l}"]) for l in layers}
    rng = np.random.default_rng(2027)
    rand = {}
    for l in layers:
        r = rng.standard_normal(len(dirs[l]))
        rand[l] = unit(r - (r @ dirs[l]) * dirs[l])
    ed = Edit(model)

    def setup_blind():
        rel = ed.direction(dirs, cols, ALPHA)
        log(f"   盲测方向 α={ALPHA}：每层相对改动 {rel}")
        return rel

    def setup_rand():
        target = ed.direction(dirs, cols, ALPHA)
        ed.restore()
        one = ed.direction(rand, cols, 1.0)
        ed.restore()
        info = {}
        for l in layers:                                   # 每层单独缩放，使改动范数与盲测方向 α=6 相同
            p = ed._w(l)
            W = p.data.float()
            u = torch.tensor(rand[l], dtype=torch.float32, device=W.device)
            mask = torch.zeros(W.shape[1], device=W.device)
            mask[torch.tensor(cols[l], device=W.device)] = 1
            new = W - (target[l] / max(one[l], 1e-12)) * torch.outer(u, mask * (u @ W))
            info[l] = ((new - W).norm() / W.norm()).item()
            p.data.copy_(new.to(p.dtype))
        log(f"   随机正交方向：每层相对改动 {info}（目标 {target}）")
        return info

    res_file = OUT / "ablate.json"
    results = json.loads(res_file.read_text("utf-8")) if res_file.exists() else {}
    for key, name, setup in (("blind_a6", f"盲测方向+盲测神经元 α={ALPHA:g}", setup_blind),
                             ("rand_dir", "对照：盲测神经元+同范数随机正交方向", setup_rand)):
        if name in results:
            log(f"[{name}] 已有结果，跳过")
            continue
        t0 = time.time()
        setup()
        outs = R.generate(model, tok, held, CAP, BATCH, SEED + 1000)
        ed.restore()
        results[name] = R.metrics(tok, outs, CAP)
        torch.save([{k: o[k] for k in ("id", "new_ids", "ended")} for o in outs], OUT / f"gen_{key}.pt")
        log(f"[{name}] {R.fmt(results[name])}  ({time.time() - t0:.0f}s)")
        res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    log(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB · 输出 {OUT}")


if __name__ == "__main__":
    main()
