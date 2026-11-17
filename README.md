# TGI-RCS — Scanning for a "loop ghost" inside a language model's weights

> 🇰🇷 [한국어](README.ko.md) | 🇨🇳 [中文](README.zh.md) | 🇺🇸 English

**이조글로벌인공지능연구소 (LEECHO Global AI Research Lab) & Claude Fable 5.2 · Anthropic** · 2026-11-15 to 2026-11-17

A two-day experiment log. We took the "direction + generator localization" idea from our TGI (Training Ghost Ideal) papers and pointed it at an open model that falls into loops, to see whether the place where the loop is written can be found from weights and hidden states **without using anyone else's answer**.

Everything is here: all data, all scripts, the results that held, the results that did not, and the mistakes we made.

---

## Summary

On one model (Qwen3.5-0.8B) and one problem (loops in thinking mode on math problems), the scan chain worked end to end:

| Question the scan must answer | Result |
|---|---|
| Is there a problem? | 53–59% of responses loop in the post-trained model; 0/16 in the pretrained-only Base model |
| When does it start? | Median loop onset around 5200–6400 tokens (almost invisible within 3072 tokens) |
| Which layers? | With the last layer excluded, layers 15 and 19 were selected independently on new problems |
| Which neurons? | 64 per layer; 65 overlap with the 128 in the published edit (about 2.3 expected by chance) |
| Is the direction stable? | Cosine 0.987 / 0.989 between directions computed from two disjoint problem sets |
| Is what we found causal? | Intervening along the scanned direction: strict loops 19/32 → 0/32. Same-size random direction: 13/32 |
| Does it hold on another problem set? | Third set: loops 15/32, layers 15 and 19 selected again; after intervention 8/32 → 0/32, random direction 10/32 |
| How many correct with enough length? | Cap 8192 → 16384 → 32768: 3 → 6 → 11 (unmodified: 3 → 3; MIT's edit: 12 → 17 → 24) |

**With enough length:** under the 8192-token cap, the number of correct answers did not increase after the loops disappeared (3/32). Raising the cap step by step to 32768 (the MIT experiment's evaluation cap) gives 3 → 6 → 11 correct on the same problems; the unmodified model is still at 3 at 16384. Under the same caps MIT's edit gives 12 → 17 → 24, and the gap does not shrink as the cap grows (9 → 11 → 13): the cap limits the number of correct answers, but it is not the main cause of the gap between ours and MIT's. See experiment I.

**What was not achieved:** our version is still below MIT's edit (17 correct already at 16384). We can find the problem, but the standard for "what a correct fix is, and how much to change" is borrowed. See [No reference answer](#no-reference-answer).

### Failures, and the boundaries they found

| Attempt | Result | Cause | Boundary it found |
|---|---|---|---|
| Version-1 metric: cap 3072, "share that hit the cap" | ❌ three conditions indistinguishable (90% / 96% / 94%) | normal thinking also hits the cap; restating the problem counted as repetition | cap of at least 8192; loop = strictly periodic tail |
| First blind test: layers self-selected, last layer included in the ranking | ❌ loops 59% → 94%, the wrong direction | the last layer was ranked on the same scale as middle layers | exclude the last layer (confirmed by experiment C, repeated in D and G) |
| Experiment A: a weight criterion to explain the last layer | ❌ the criterion did not discriminate | the direction leans toward the same subspace in every layer | a threshold must require "more than the control group" |
| Experiment B: locating by the weight difference before/after post-training | ❌ overlap with the published edit 1/128 | post-training changed 96.6% of the weights, diffusely | weights alone cannot locate it; the model must show the problem first |
| Experiment D: zeroing the selected neurons | ⚠️ localization holds, causality inconclusive (p=0.13) | zeroing is not selective, normal reasoning is damaged too | the location is right; the intervention must change |
| Subtracting the direction, dose 1 | ❌ not significant in any of the three layers | MIT's edit was taken to be dose 1; it is 6 | dose is the key (experiments E, F: loops 19/32 → 0/32) |
| Comparing correct answers under the 8192 cap | ❌ 3 vs 3, no visible difference | the cap cut off responses that would have finished | align the cap to 32768 (experiment I: 3 → 6 → 11) |

✅ What worked is in the table above. Every failure in this table led to one of the conclusions that later held.

---

## Background

**The TGI papers.** In May 2026, starting from the "goblin" behaviour reported in GPT models, we wrote four papers (see [`papers/`](papers/), in Chinese) that compare "a structure left in the weights by training, which keeps absorbing the output once triggered" to an ideal in ring theory. The papers give definitions and metric formulas. None of those formulas had been computed on a real model.

**The object we scanned.** In November 2026, a public experiment — *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights* by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab of MIT CSAIL ([post](https://rikiyat.github.io/blog/weight-space-programming), published 2026-11-02; called "the MIT experiment" here; authors and affiliation checked against the post on 2026-11-17) — had an LLM agent (Astra) rewrite the weights of Qwen3.5-0.8B directly: 64 columns of the MLP output matrix in each of layers 15 and 19. Loops on math problems dropped sharply. The logs are public at [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math`).

This gave us a rare setting: **a model known to have a problem, plus someone else's published location of that problem.** So we could test whether the TGI method, without looking at that answer, finds the same place.

**Related work from the same week.** Will DePue's [Checkpoint Forensics](https://unemploymentlabs.com/checkpoint-forensics) (2026-11-12) infers from published weights which GPU class generated the initial weights. It reads initialization traces that training did not erase and says nothing about model behaviour, but it is another case of information being readable from weights.

---

## Setup

- **Models:** `Qwen/Qwen3.5-0.8B` (post-trained, revision `2fc06364715b967f1860aea9cf38778875588b17`, weight hashes match the MIT experiment's baseline) and `Qwen/Qwen3.5-0.8B-Base` (pretrained only). 24 layers, 3584 MLP neurons per layer, residual width 1024.
- **Generation:** thinking mode, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 (the MIT experiment's evaluation settings), cap 8192 tokens.
- **Problems:** the MIT experiment's public practice set `math_train.jsonl`, 32 problems per condition, same random seed across conditions.
- **Loop criterion:** the tail of the response is strictly periodic (≥90% of the last max(4p,160) tokens satisfy `ids[t]==ids[t-p]`).
- **"Neuron":** one dimension of the MLP hidden layer, i.e. the value entering `mlp.down_proj`.
- **Decision thresholds were written before each run** (stated per section). Post-hoc analyses are marked as such.
- **Hardware:** a single NVIDIA DGX Spark (unified memory). Peak GPU memory actually used during generation was about 7.5–11.5 GB; what the process holds from the system grows up to the cap that is set (we set 18–24 GB).

---

## Experiment log (chronological)

### Day 1 (2026-11-15)

**0. Assessment of our existing software.** The six metric functions in our own TGI Scanner (a GTK application, not open-sourced) turned out to be statistics on output text. They implement none of the mathematical objects in the papers. This was the starting point.

**1. Internal probe (measurement only).** `rcs/tgi_probe.py`

| | Qwen3.5-0.8B-Base | Qwen3.5-4B-Base |
|---|---|---|
| Exact entropy μ / σ (bits) | 1.896 / 2.170 | 1.187 / 1.653 |
| μ − 2σ | −2.445 (0 steps below) | −2.119 (0 steps below) |
| Entropy: code / math / literature / medicine | 1.25 / 0.79 / 3.65 / 1.94 | 0.69 / 0.48 / 2.76 / 1.13 |

Problem found: the entropy-sentinel threshold μ−kσ from the paper is negative and can never fire; entropy also differs widely by domain.

**2. Testing the MIT experiment, version 1 (failed).** Cap 3072 tokens. The "hit the cap" rate could not separate the three conditions (90% / 96% / 94%). The metric was wrong: at 3072 tokens normal thinking also hits the cap, and "8-gram seen before" counts restating the problem as repetition. **This was pointed out by the author without reading the code.**

**3. Version 2 (cap 8192, loop = periodic tail).** `rcs/tgi_rcs.py`, data in `data/rcs_mit_Qwen3.5-0.8B_v2/`

The MIT layer numbers (15, 19) were used here; its neuron indices were not:

| Ranking (top 64 in each of the two layers) | Overlap with MIT's 128 | Pre-registered | Result |
|---|---|---|---|
| Activation difference, in-loop − normal | 19 | ≥16 | holds |
| Amount written into the ghost direction | 68 | ≥32 | holds |
| Domain candidate set (math top-1000) | 1 | — | recorded only |

| Condition (32 held-out problems) | Loops | Answers given |
|---|---|---|
| Unmodified | 17/32 (53%) | 5 |
| MIT's actual edit | 1/32 (3%) | 10 |
| Zero MIT's 128 neurons | 4/32 (12%) | 7 |
| Zero top-64 writers | 8/32 (25%) | 2 |
| Zero top-64 by activation difference | 11/32 (34%) | 3 |
| Zero 64 random neurons, same layers | 16/32 (50%) | 6 |

**4. First blind test (did not hold).** Without the MIT layer numbers, layers were chosen by "sum of the top-64 write amounts per layer". This picked layers 23 and 19.

| Condition | Loops |
|---|---|
| Random zeroing, same layers | 19/32 (59%) |
| Zero the 128 self-selected neurons | **30/32 (94%)** |
| Subtract the direction from them (dose 1) | 13/32 (41%) |

The effect went the wrong way: more loops, starting earlier. So the success in step 3 depended on the layer numbers MIT supplied.

**5. Base control.** Same settings, pretrained-only Base model: 0/16 loops, 12/16 answers given. The loop appears after post-training.

### Day 2 (2026-11-16)

**A. Are the last-layer neurons "confidence-regulation neurons"? (test design was not discriminative)** `rcs/tgi_weights.py`

Criterion: the fraction of a neuron's output weight lying in the near-null space of the unembedding. 27 of the 64 self-selected layer-23 neurons were in that layer's top 5% (3.2 expected), which "holds" by the pre-registered threshold. **But the groups in layers 15 and 19 were just as enriched (21–31).** The criterion cannot single out the last layer. Post-hoc reason: the ghost direction itself leans toward the near-null space in every layer (fraction 0.13–0.44, random 0.05).

**B. Can the weight difference between Base and post-trained locate the loop? (does not hold)**

- Post-training changed 96.6% of the language model's weight values, each by about 5% on average.
- MLP output matrices changed by 6.8–10.5% per layer, with no layer standing out.
- Alignment of the weight difference with the ghost direction: 1.05–1.58 in every layer (1 = random).
- The 64 most-changed neurons overlap with MIT's 128 in 1 neuron.

For scale: the MIT edit touched 131,072 weights, about 0.017% of the language model.

**C. Each layer on its own.** `rcs/tgi_layers.py`, data in `data/rcs_layers_20261116/` (baseline 17/32)

| Layer | Zero 64 neurons | Subtract direction (dose 1) |
|---|---|---|
| 15 | 5/32 (16%) | 13/32 (41%) |
| 19 | 20/32 (62%) | 13/32 (41%) |
| 23 | 27/32 (84%) | 16/32 (50%) |

Holds by the pre-registered threshold: the failure of the first blind test came from the last layer. All of the benefit of zeroing came from layer 15.

**D. Second blind test (new problems, rule fixed in advance).** `rcs/tgi_blind2.py`, data in `data/rcs_blind2_20261116/`

Rule: layer score S(l) = d_l · W_l (a_loop − a_normal), **last layer excluded**, take the top 2 layers; in each, the 64 neurons with the largest write. Problems 201–232 for selection, 233–264 for evaluation. No MIT layer numbers or neuron indices.

- Selected **layers 19 and 15** (the last layer scored second and would have been picked if not excluded).
- 65 of the selected neurons overlap with MIT's 128.

| Condition | Loops | Answers given |
|---|---|---|
| Unmodified | 19/32 (59%) | 3 |
| Zero the 128 self-selected neurons | 6/32 (19%) | 4 |
| Random zeroing, same layers | 11/32 (34%) | 8 |
| MIT's actual edit | 1/32 (3%) | 12 |

Localization holds. The causal part is **inconclusive** by the pre-registered threshold (≥20 points below the random control): 15.6 points, p=0.13; against unmodified, p=0.0009.

**E. Causal verification: paired texts + log-probabilities.** `rcs/tgi_pairs.py`, data in `data/pairs_20261116/`

Reading the MIT experiment's raw logs showed that its final formula is `W = W0 − 6·outer(u, mask·(uᵀW0))`. **The dose is 6**: the write goes from c to −5c. We had been comparing against dose 1 all along.

Following its method, six hand-written text pairs (same problem, same opening, one productive check and one repeating the same sentence) were used to measure how much the mean log-probability of the repetitive branch and of the productive branch changes after an edit.

| Condition | Repetitive | Productive |
|---|---|---|
| Calibration: MIT's actual edit (self-reported −0.103 / −0.009) | −0.101 | −0.006 |
| Our direction + our neurons, dose 1 | −0.015 | +0.001 |
| Same, dose 2 / 4 | −0.031 / −0.061 | −0.001 / −0.008 |
| Same, dose 6 | −0.092 | −0.014 |
| Blind-selected direction + neurons, dose 6 | −0.082 | −0.015 |
| Our neurons + MIT's direction, dose 6 | −0.088 | −0.008 |
| Zero our 128 neurons | −0.073 | −0.064 |
| Same-size random orthogonal direction | −0.004 | −0.017 |

- Cosine between our direction and MIT's: 0.67 (layer 15) / 0.71 (layer 19).
- The earlier weak effect of "subtracting the direction" was a dose problem, not a direction problem.
- Zeroing is not selective: repetitive and productive branches drop together.

**F. Confirmation by generation.** `rcs/tgi_dose.py`, data in `data/rcs_dose_20261116/`

Direction and neurons selected in D, dose 6, generated on D's evaluation problems:

| Condition | Loops | Answers given | Hit the cap |
|---|---|---|---|
| Unmodified | 19/32 (59%) | 3 | 29 |
| **Blind direction + blind neurons, dose 6** | **0/32** | 3 | 29 |
| Same neurons + same-size random direction | 13/32 (41%) | 2 | 30 |
| MIT's actual edit | 1/32 (3%) | 12 | 20 |

Holds by the pre-registered threshold (41 points lower, p=0.00003). **But under the 8192 cap the number of answers did not increase:** responses no longer repeat periodically, yet they keep re-checking with "Wait, …" over and over until the cap. For what happens with a higher cap, see experiment I.

**G. Replication on a third problem set.** Data in `data/scan_e2e_20261116/`

The procedures of D and F were run as one pass (layer selection, neuron selection, three verification conditions): problems 265–296 for scanning, 297–328 for verification. This run used a consolidated script that is part of our non-open-sourced TGI Scanner; the algorithm is the same as `tgi_blind2.py` + `tgi_dose.py`.

- Loops 15/32, median onset 5872 tokens.
- Layers 19 and 15 were selected again; the last layer again ranked second.

| Condition | Loops | Finished |
|---|---|---|
| Unmodified | 8/32 | 3 |
| Scanned direction, dose 6 | **0/32** | 6 |
| Same-size random direction | 10/32 | 4 |

31 points lower, p=0.0004. The same conclusion on three disjoint problem sets.

**H. Grading against the reference answers.** `rcs/tgi_grade.py`, data in `data/grade_20261116/`

The problem set comes with reference answers. The content of the last `\boxed` after `</think>` is normalized and must match the reference exactly (conservative; a manual check found 3 false negatives).

| Condition (D's evaluation problems, cap 8192) | Correct |
|---|---|
| Unmodified | 3/32 |
| Blind direction + blind neurons, dose 6 | 3/32 |
| Zero the 128 self-selected neurons | 4/32 |
| Random zeroing, same layers | 8/32 |
| Same-size random direction | 2/32 |
| MIT's actual edit | 12/32 |

- When the post-trained model finishes, it is almost always correct. On this model, "how many correct" is essentially "how many finished within the cap".
- The Base model got 11 of 16 correct (after the manual check); the post-trained model got 2 of problems 1–32 under the same settings.

**I. Progressive token-cap test.** `rcs/tgi_continue.py`, data in `data/continue_20261116/` and `data/continue32k_20261116/`

Question: does the 8192 cap cut off responses that would have finished? The MIT experiment evaluates at 32768 and reports a median length of 9171 tokens for correct responses after its edit.

Method: for each condition, apply the corresponding weight edit and continue the responses that were cut at 8192, first to 16384, then to 32768. The earlier tokens are unchanged; the continuation uses a new random seed.

| Condition (D's evaluation problems, 32) | Correct @8192 | @16384 | @32768 |
|---|---|---|---|
| Unmodified | 3 | 3 | not run (28 of the 29 unfinished responses are already in strict loops) |
| Blind direction + blind neurons, dose 6 | 3 | 6 | **11** |
| MIT's actual edit | 12 | 17 | **24** |

Detail for our version:

| Cap | Finished | Still unfinished | Of which strict loops |
|---|---|---|---|
| 8192 | 3 | 29 | 0 |
| 16384 | 6 | 26 | 2 |
| 32768 | 13 | 19 | 4 |

- Each doubling of the cap raised the number of correct answers. Of the 7 responses that finished between 16384 and 32768, 5 finished after 27000 tokens.
- More length does nothing for the unmodified model: after continuation, 28 of 29 are in strict loops. This is direct data for "once the loop starts, it persists".
- At 32768 there was 1 response that finished with a wrong answer and 1 that finished without giving an answer.
- Among responses still unfinished at 16384, the median count of "Wait" in the last 2000 tokens: unmodified 52, ours 44, MIT's 14. Our edit suppressed verbatim repetition but not the repeated self-doubt.
- Pre-registered: at 16384 ours is 3 above unmodified (threshold: 7 or more holds, fewer than 3 does not) → inconclusive. The judgement at 32768 (gap to MIT ≤5 means the cap is the main cause of the gap, ≥9 means it is not): the gap is 13 (11 vs 24) → **the cap is not the main cause of the gap**.
- The gap to MIT grows as the cap is raised: 9 (@8192) → 11 (@16384) → 13 (@32768).
- MIT's edit at 32768 (run on 2026-11-17): 27 finished, 25 gave an answer, 24 correct; of the 5 unfinished, 3 are strict loops. Of the 10 responses that finished between 16384 and 32768, 4 finished after 27000 tokens.

**How to read this comparison**

- **The token-cap boundary has to be found first.** Any method that uses a model's output as its standard of comparison (TGI included) must first find the threshold for "how long does a response need in order to finish", and only then compare. This is a prerequisite analysis, not an optional one. In this experiment 8192 gives 3 vs 3, no visible difference; the conditions only separate at 32768.
- **MIT's edit is the result of many rounds; ours is a single round.** MIT's run (p50c) used 50 forward passes, 158 turns, 80 Python runs and 20.3 hours; of the 50 counted passes, 33 were paired comparisons of the original and the edited model on the same problems, each returning accuracy, and 16 were reads of internal activations (see its published `passes.md`). Our blind direction and blind neurons come from one scan and one intervention, with no accuracy feedback used to adjust the edit (the only manual rule added on the way was "exclude the last layer"; the size and dose are borrowed from MIT).
- **What is verified here is whether the theory and the mechanism hold, not the limit of what this system can reach.** The 24 vs 11 gap should not be read as the ceiling of the method.

Correctness of the continuation: compared with a full forward pass, the next-token logits from the cached continuation differ by about 0.2 (bf16 noise, the same size as "batched" vs "single"), with the same top-1 and top-5.

---

## What was not shown, and limits

- **One model, one problem.** Qwen3.5-0.8B, loops in thinking mode on math problems.
- **Small samples.** 32 problems per condition.
- **"Exclude the last layer" is a rule added by hand after a failure**, not something the method derives. What those last-layer neurons do was not answered.
- **Scale and dose are borrowed.** "2 layers × 64 neurons" and dose 6 both come from the MIT experiment.
- **The model must loop before it can be scanned.** Weights alone do not reveal it (experiment B).
- **The loop criterion measures strict periodic loops only**, not whether the model finishes.
- **The grader is conservative**: correct answers written differently are marked wrong (3 found by a manual check).
- **The 32768 results are stitched from two continuations:** each continuation uses a new random seed, so it is not a single run; the original MIT experiment generates to 32768 in one pass.
- **Generation on day 1 and in experiments C, D and F only went to 8192**, so the "answers given" counts in those tables are truncated by the cap.
- **Of the six metrics in the papers, only loop onset was realized.** The ring-theoretic formalization is not validated by these experiments.
- Each random control is a single draw.

---

## Mistakes we made

They are listed because they are as much a fact of these two days as the results.

1. Version-1 metric was invalid: cap too short, and "8-gram repetition" counted restatement.
2. The first blind test ranked the last layer on the same scale as middle layers. The relevant literature (confidence-regulating neurons in the last layer) was known and was not used in the design.
3. The threshold for experiment A did not require "more enriched than the control layers". It passed literally and discriminated nothing.
4. MIT's edit was treated as "subtract the direction" (dose 1). It is dose 6, and the formula is written in its final report.
5. The goal drifted for a while: TGI is a system for scanning problems, yet we started judging by "how many problems are answered after the fix".
6. A provisional idea (candidate sets by domain) was tested as if it were a prediction of the theory.
7. One run computed full-vocabulary logits, hit the memory cap, and lost 28 minutes of results.
8. When comparing with MIT's results, the token cap was not aligned with its 32768 from the start. 8192 was used to save time, which made the comparison of "correct answers" invalid.
9. Memory was reported on the wrong basis: "peak about 7.5 GB" (memory in use) while the process was holding about 23 GB from the system. On a unified-memory machine that difference matters.

---

## No reference answer

Models are still black boxes. All three of these efforts first find something certain outside the model to anchor on, then reason back to the weights:

| | External anchor | What it touches | What it cannot touch |
|---|---|---|---|
| Checkpoint Forensics | Determinism of the random-number generator | Which chip class generated the initial weights | What training learned |
| The MIT experiment | Correct answers to math problems | Which neurons to change to answer more problems | Why those weights are wrong |
| This repository | Behavioural contrast, looping vs normal | Which layers, which neurons, which direction the loop is written in | What a correct fix is, and how much to change |

None of them can go forward from weights to "what will happen". To move from scanning to correction, one needs a reference standard that tells right weights from wrong ones. We do not have it.

We are blind men feeling an elephant. But some of what we touched is real. All the data is public. Keep feeling.

---

## Layout

```
papers/     four TGI papers (HTML, Chinese)
rcs/        experiment scripts (comments and logs are in Chinese)
data/       all results: JSON, logs, statistics (npz)
```

| Script | What it does | Data |
|---|---|---|
| `tgi_probe.py` | exact entropy, attention Gini, domain neurons | `data/probe_*` |
| `tgi_rcs.py` | reproduce loops, ranking, zero-ablation, first blind test | `data/rcs_mit_*`, `data/rcs_blind_*` |
| `tgi_weights.py`, `elem_changed.py`, `posthoc_ghost_rho.py` | experiments A and B (weights only) | `data/weights_20261116/` |
| `tgi_layers.py` | experiment C | `data/rcs_layers_20261116/` |
| `tgi_blind2.py` | experiment D | `data/rcs_blind2_20261116/` |
| `tgi_pairs.py` | experiment E | `data/pairs_20261116/` |
| `tgi_dose.py` | experiment F | `data/rcs_dose_20261116/` |
| (consolidated script, not open-sourced) | experiment G | `data/scan_e2e_20261116/` |
| `tgi_grade.py` | experiment H: grading against reference answers | `data/grade_20261116/` |
| `tgi_continue.py` | experiment I: continuing to a higher cap | `data/continue_20261116/`, `data/continue32k_20261116/` |

`data/` includes the token sequences of the generations (`.pt`, lists saved with PyTorch). Only the raw activations of the 4B model are left out.

## Reproducing

Requirements: Python 3.10+, PyTorch, a transformers version that supports Qwen3.5, safetensors, numpy, and a GPU (about 8 GB at cap 8192; for the continuation to 32768 we set the process cap to 18 GB).

```bash
# 1. Models
#    Qwen/Qwen3.5-0.8B (revision 2fc06364715b967f1860aea9cf38778875588b17) and Qwen/Qwen3.5-0.8B-Base
export TGI_MODEL_DIR=/path/to/Qwen3.5-0.8B
export TGI_BASE_DIR=/path/to/Qwen3.5-0.8B-Base

# 2. Public files of the MIT experiment (not redistributed here), placed in Ref/mit_wsp_exp8/
#    From exp8_math in https://github.com/RikiyaT/lm_understand-logs :
#    edit.npz, manifest.json (runs/p50c/edit), math_train.jsonl (practice),
#    p50c_logs/tool_calls.jsonl (logs of runs/p50c)

# 3. Runs that finish in minutes
cd rcs
python tgi_weights.py     # experiments A and B, CPU only, about 10 s, needs about 10 GB RAM
python tgi_pairs.py       # experiment E, about 1 minute

# 4. Generation experiments, about 19 minutes per condition
python tgi_blind2.py      # experiment D, about 95 minutes
python tgi_dose.py        # experiment F, about 40 minutes (needs D's output)

# 5. Grading and continuation
python tgi_grade.py       # experiment H, CPU only, about 1 minute
python tgi_continue.py --extra 8192      # experiment I, to 16384, about 30–40 minutes per condition
python tgi_continue.py --extra 16384 --conds blind_a6,mit --src ../data/continue_20261116 \
    --out ../data/continue32k_20261116 --batch 13 --max-mem-gb 18      # to 32768, about 75 minutes per batch
```

Before publishing we re-ran `tgi_weights.py`, `tgi_pairs.py` and `tgi_grade.py` on a copy of this directory; the numbers match `data/`. The other scripts had only their paths changed and were not re-run end to end.

## Papers

| File | Title |
|---|---|
| [`01_ring_ideal_isomorphism.zh.html`](papers/01_ring_ideal_isomorphism.zh.html) | Isomorphic Mapping Between Ring Ideals and Attractor Dynamics in Neural Weight Space |
| [`02_ideals_in_attention_layers.zh.html`](papers/02_ideals_in_attention_layers.zh.html) | On the Distribution of Ring-Theoretic "Ideals" in LLM Attention Layers |
| [`03_tgi_engineering_framework.zh.html`](papers/03_tgi_engineering_framework.zh.html) | Training Ghost Ideal — Full-Stack Engineering Framework |
| [`04_scanner_v1_error_report.zh.html`](papers/04_scanner_v1_error_report.zh.html) | TGI Scanner v1.0 — Experimental Error Rates, Known Limitations, and Failure Modes |

The papers were written in May 2026, before the experiments in this repository. The experiments test the "direction + generator localization" part, not the whole theory.

## How these experiments were done

The research question, the theory and the judgement at each step come from this lab. The design, coding and execution of the experiments were done in collaboration with the AI model Claude Fable 5.2 (Anthropic, used through Claude Code), running on our own machine. As in our papers, the collaborating AI model is named. Most of the items under "Mistakes we made" actually happened in that collaboration.

## License and acknowledgements

- License: [Apache License 2.0](LICENSE). Any secondary development is welcome.
- The scanned experiment: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL), [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming), Experiment 8; logs at [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math`). We thank the authors for publishing the complete logs. Its files are not redistributed here; `data/` contains excerpts of what the model generated on its practice problems.
- Models: Qwen3.5-0.8B and its Base version (Alibaba Qwen team).
- Confidence-regulation neurons: Stolfo et al., 2024.

## Contact

이조글로벌인공지능연구소 (LEECHO Global AI Research Lab) · Incheon, South Korea · [leechoglobalai.com](https://leechoglobalai.com)
