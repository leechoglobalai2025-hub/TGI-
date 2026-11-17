#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 判分（2026-11-16）：给已保存的生成结果按标准答案判对错。不做新的生成，只用 CPU。

判分规则（保守，可能把写法不同的正确答案判成错）：
  取 </think> 之后最后一个 \\boxed{...} 的内容，与题库里的 answer 都做同样的归一化后比较是否完全相同。
  归一化：去空白、去 $、去 \\left \\right \\! \\, 、\\dfrac/\\tfrac → \\frac、去 \\text{..} 的外壳、
          去末尾句点、去千分位逗号、去 ^\\circ、去 \\% 和 %、"x=" 之类的前缀只保留等号右边。
  没有 </think> 或没有 \\boxed 的算"未作答"。

用法：python tgi_grade.py
"""
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tgi_rcs as R  # noqa: E402

EXPORT = R.ROOT / "data"
MODEL = os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B")


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def norm(s):
    s = s.strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(left|right|!|,|;|:| )", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("$", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
    if "=" in s and not s.startswith("("):
        s = s.split("=")[-1]
    return s.rstrip(".")


def grade(tok, outs, answers):
    n = len(outs)
    fin = boxed = correct = 0
    rows = []
    for o in outs:
        text = tok.decode(o["new_ids"], skip_special_tokens=False)
        ans = last_boxed(text.split("</think>")[-1]) if "</think>" in text else None
        ok = ans is not None and norm(ans) == norm(answers[o["id"]])
        fin += bool(o["ended"])
        boxed += ans is not None
        correct += ok
        rows.append({"id": o["id"], "finished": bool(o["ended"]), "given": ans, "truth": answers[o["id"]], "correct": ok})
    return {"n": n, "finished": fin, "boxed": boxed, "correct": correct, "rows": rows}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    answers = {json.loads(x)["id"]: json.loads(x)["answer"] for x in open(R.REF / "math_train.jsonl", encoding="utf-8")}
    sets = [
        ("D 评测题（第 233–264 题）", [
            ("不改动", "rcs_blind2_20261116/gen_1.pt"),
            ("盲测：置零自选 128 个", "rcs_blind2_20261116/gen_2.pt"),
            ("对照：同层随机置零", "rcs_blind2_20261116/gen_3.pt"),
            ("参照：MIT 的实际改动", "rcs_blind2_20261116/gen_4.pt"),
            ("盲测方向+盲测神经元 α=6", "rcs_dose_20261116/gen_blind_a6.pt"),
            ("对照：同范数随机正交方向", "rcs_dose_20261116/gen_rand_dir.pt")]),
        ("C 的题（第 33–64 题；昨天的不改动和 MIT 条件没有保存生成结果）", [
            ("只置零第 15 层", "rcs_layers_20261116/gen_z15.pt"),
            ("只置零第 19 层", "rcs_layers_20261116/gen_z19.pt"),
            ("只置零第 23 层", "rcs_layers_20261116/gen_z23.pt"),
            ("第 15 层减方向 α=1", "rcs_layers_20261116/gen_p15.pt"),
            ("第 19 层减方向 α=1", "rcs_layers_20261116/gen_p19.pt"),
            ("第 23 层减方向 α=1", "rcs_layers_20261116/gen_p23.pt")]),
        ("采集用的题（未修改模型）", [
            ("第 1–32 题（昨天）", "rcs_mit_Qwen3.5-0.8B_v2/collect_ids.pt"),
            ("第 201–232 题（D）", "rcs_blind2_20261116/collect_ids.pt"),
            ("Base 版，第 1–16 题", "rcs_blind_Qwen3.5-0.8B-Base/collect_ids.pt")]),
    ]
    out = {}
    for title, conds in sets:
        print(f"\n== {title} ==")
        for name, rel in conds:
            p = EXPORT / rel
            if not p.exists():
                print(f"  {name}: 文件不存在 {rel}")
                continue
            g = grade(tok, torch.load(p), answers)
            out[f"{title} | {name}"] = g
            print(f"  {name:<28} 答对 {g['correct']:>2}/{g['n']}   给出 boxed {g['boxed']:>2}   结束 {g['finished']:>2}")
    d = EXPORT / "grade_20261116"
    d.mkdir(exist_ok=True)
    (d / "grade.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    wrong = [(k, r) for k, g in out.items() for r in g["rows"] if r["given"] is not None and not r["correct"]]
    print(f"\n给出答案但判为错的共 {len(wrong)} 条，前 12 条（给出 → 标准）：")
    for k, r in wrong[:12]:
        print(f"  [{k.split('|')[-1].strip()[:14]}] {r['given'][:40]!r} → {r['truth'][:40]!r}")
    print(f"\n输出 {d / 'grade.json'}")


if __name__ == "__main__":
    main()
