#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGI 续写测试（2026-11-16）：把在 8192 token 处被截断的回答接着往下生成，检验"token 上限是否卡住了答对数"。
规则与判定标准预先写在 SESSION_20261116_TGI_Weights.md。

做法：对每个条件装上对应的权重改动；把"提示 + 已生成的 8192 个 token"分块前向一遍建立缓存
（只算隐状态，不算整段的词表 logits）；再用与原来相同的采样设置续写。前 8192 个 token 不变。
续写用新的随机种子，统计上等价于一开始就给更高的上限，但不是逐字相同的同一次运行。

用法：
  python tgi_continue.py --selftest          # 不采样时，续写 与 一次写完 应逐字一致
  python tgi_continue.py --extra 8192
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
from tgi_grade import grade  # noqa: E402
from tgi_pairs import Edit, unit  # noqa: E402

EXPORT = R.ROOT / "data"
B2 = EXPORT / "rcs_blind2_20261116"
OUT = EXPORT / "continue_20261116"
MODEL = os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B")
log = R.log


def prompt_ids(tok, problem):
    text = tok.apply_chat_template([{"role": "user", "content": problem + R.SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def prefill(model, ids, att, chunk=1024):
    """分块前向建立缓存，只走主干（不算词表 logits）。留最后一个 token 给 generate。"""
    past = None
    T = ids.shape[1] - 1
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        out = model.model(input_ids=ids[:, s:e], attention_mask=att[:, :e], past_key_values=past, use_cache=True)
        past = out.past_key_values
    return past


@torch.no_grad()
def continue_batch(model, tok, seqs, n_prefix_gen, extra, seed, sample=True, batch=32):
    """seqs: [(提示 ids, 已生成 ids)]，每条已生成长度都是 n_prefix_gen。返回每条新增的 token 和是否结束。"""
    from transformers import LogitsProcessorList
    stop = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    res = []
    for k in range(0, len(seqs), batch):
        chunk = seqs[k:k + batch]
        T = max(len(p) + len(g) for p, g in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, (p, g) in enumerate(chunk):
            seq = p + g
            ids[j, T - len(seq):] = torch.tensor(seq)          # 左填充
            att[j, T - len(seq):] = 1
        ids, att = ids.cuda(), att.cuda()
        past = prefill(model, ids, att)
        torch.manual_seed(seed + k)
        kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                  logits_processor=LogitsProcessorList([R.make_presence(1.5, T - n_prefix_gen)])) if sample \
            else dict(do_sample=False)
        out = model.generate(input_ids=ids, attention_mask=att, past_key_values=past, max_new_tokens=extra,
                             eos_token_id=stop, pad_token_id=pad, **kw)
        for j in range(len(chunk)):
            new = out[j, T:].tolist()
            ended = False
            for i, t in enumerate(new):
                if t in stop:
                    new, ended = new[:i], True
                    break
            res.append((new, ended))
        del past, out
        torch.cuda.empty_cache()                               # 归还缓存，统一内存主机上必须做
    return res


@torch.no_grad()
def selftest(model, tok, problems):
    """不采样：一次写 N 个 token，与"先写 n1 个、再续写 N−n1 个"应逐字一致（两条长度不同的提示，左填充）。"""
    stop = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.padding_side = "left"
    ps = [prompt_ids(tok, problems[0]["problem"]), prompt_ids(tok, problems[3]["problem"])]
    N, n1 = 1400, 1100                                       # n1 > 1024，分块前向会跨块
    T = max(len(p) for p in ps)
    ids = torch.full((2, T), pad, dtype=torch.long)
    att = torch.zeros((2, T), dtype=torch.long)
    for j, p in enumerate(ps):
        ids[j, T - len(p):] = torch.tensor(p)
        att[j, T - len(p):] = 1
    full = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=N, do_sample=False,
                          eos_token_id=stop, pad_token_id=pad)[:, T:].tolist()
    seqs = [(ps[j], full[j][:n1]) for j in range(2)]
    cont = continue_batch(model, tok, seqs, n1, N - n1, 0, sample=False)
    for j in range(2):
        a, b = full[j][n1:], cont[j][0]
        b_full = b + ([] if len(b) == len(a) else [])
        same = sum(x == y for x, y in zip(a, b_full))
        log(f"  第 {j} 条（提示 {len(ps[j])} token）：一次写完的后 {len(a)} 个 与 续写的 {len(b)} 个，逐位相同 {same}/{len(a)}；"
            f"首个不同位置 {next((i for i, (x, y) in enumerate(zip(a, b_full)) if x != y), None)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--extra", type=int, default=8192)
    ap.add_argument("--conds", default="base,blind_a6,mit")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--src", default="", help="从这个目录的 gen_<条件>.pt 接着续写（上一轮续写的输出）；不给则用最初 8192 的结果")
    ap.add_argument("--out", default="", help="输出目录")
    ap.add_argument("--batch", type=int, default=32, help="每批续写多少条；每批结束立即存盘")
    ap.add_argument("--max-mem-gb", type=float, default=20)
    args = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_mem_gb / total))
    model, tok = R.load_model(MODEL)
    tok.padding_side = "left"
    problems = [json.loads(x) for x in open(R.REF / "math_train.jsonl", encoding="utf-8")]
    if args.selftest:
        selftest(model, tok, problems)
        log(f"峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        return
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {p["id"]: p for p in problems}
    answers = {p["id"]: p["answer"] for p in problems}
    sel = json.loads((B2 / "selection.json").read_text("utf-8"))
    cols = {l: sel["cols"][str(l)] for l in (15, 19)}
    d2 = np.load(B2 / "dirs.npz")
    dirs = {l: unit(d2[f"L{l}"]) for l in (15, 19)}
    ed = Edit(model)
    conds = {"base": ("不改动", B2 / "gen_1.pt", lambda: None),
             "blind_a6": ("盲测方向+盲测神经元 α=6", EXPORT / "rcs_dose_20261116" / "gen_blind_a6.pt",
                          lambda: ed.direction(dirs, cols, 6.0)),
             "mit": ("参照：MIT 的实际改动", B2 / "gen_4.pt", lambda: ed.set(R.mit_edited_weights(model)))}
    res_file = out_dir / "continue.json"
    results = json.loads(res_file.read_text("utf-8")) if res_file.exists() else {}
    for key in [c for c in args.conds.split(",") if c]:
        name, path, setup = conds[key]
        if key in results:
            log(f"[{name}] 已有结果，跳过")
            continue
        if args.src:
            path = Path(args.src) / f"gen_{key}.pt"
        outs = torch.load(path)
        before = grade(tok, outs, answers)
        todo = [o for o in outs if not o["ended"]]
        n_gen = len(todo[0]["new_ids"])
        assert all(len(o["new_ids"]) == n_gen for o in todo)
        t0 = time.time()
        setup()
        cont = []
        for bi, k in enumerate(range(0, len(todo), args.batch)):      # 一批一批跑，每批结束立即存盘
            part = todo[k:k + args.batch]
            tb = time.time()
            c = continue_batch(model, tok, [(prompt_ids(tok, by_id[o["id"]]["problem"]), o["new_ids"]) for o in part],
                               n_gen, args.extra, args.seed + 1000 * bi, batch=len(part))
            cont += c
            pm = [{"id": o["id"], "new_ids": o["new_ids"] + new, "ended": ended} for o, (new, ended) in zip(part, c)]
            torch.save(pm, out_dir / f"gen_{key}_batch{bi + 1}.pt")
            pg = grade(tok, pm, answers)
            log(f"   [{name}] 第 {bi + 1} 批 {len(part)} 条：新结束 {pg['finished']} · 答对 {pg['correct']} · "
                f"结束长度 {sorted(len(x['new_ids']) for x in pm if x['ended'])}  ({time.time() - tb:.0f}s；"
                f"显存 已分配峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} / 已保留 {torch.cuda.memory_reserved() / 2**30:.1f} GiB)")
        ed.restore()
        merged, it = [], iter(cont)
        for o in outs:
            if o["ended"]:
                merged.append(dict(o))
            else:
                new, ended = next(it)
                merged.append({"id": o["id"], "new_ids": o["new_ids"] + new, "ended": ended})
        after = grade(tok, merged, answers)
        m = R.metrics(tok, merged, n_gen + args.extra)
        torch.save(merged, out_dir / f"gen_{key}.pt")
        results[key] = {"name": name, "cap_before": n_gen, "cap_after": n_gen + args.extra, "continued": len(todo),
                        "before": {k: before[k] for k in ("n", "finished", "boxed", "correct")},
                        "after": {k: after[k] for k in ("n", "finished", "boxed", "correct")},
                        "loops_after": round(m["hard_loop"] * m["n"]), "hit_cap_after": round(m["hit_cap"] * m["n"]),
                        "lengths_finished": [len(x["new_ids"]) for x in merged if x["ended"]],
                        "seconds": time.time() - t0}
        res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
        log(f"[{name}] 续写 {len(todo)} 条：答对 {before['correct']} → {after['correct']}/{after['n']} · "
            f"结束 {before['finished']} → {after['finished']} · 续写后陷入循环 {results[key]['loops_after']} · "
            f"撑满新上限 {results[key]['hit_cap_after']}  ({time.time() - t0:.0f}s)")
    log(f"峰值显存 已分配 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB / 已保留 {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB · 输出 {out_dir}")


if __name__ == "__main__":
    main()
