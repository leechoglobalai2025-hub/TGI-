# 2026-11-16：逐元素统计 Base → 后训练有多少权重的值不同（BF16 存储值逐位比较）
import os
import torch, collections
from safetensors.torch import load_file
d=os.environ.get("TGI_MODELS", "models")+"/%s/model.safetensors-00001-of-00001.safetensors"
p,b=load_file(d%"Qwen3.5-0.8B"),load_file(d%"Qwen3.5-0.8B-Base")
g=collections.defaultdict(lambda:[0,0,0.0,0.0]); same=[]
for k in p:
    part="语言模型" if k.startswith("model.language_model.") else ("视觉" if k.startswith("model.visual.") else "mtp")
    ne=(p[k]!=b[k]); n=ne.sum().item()
    if n==0: same.append((k,tuple(p[k].shape)))
    x=g[part]; x[0]+=n; x[1]+=p[k].numel()
    x[2]+=(p[k].double()-b[k].double()).pow(2).sum().item(); x[3]+=b[k].double().pow(2).sum().item()
    if k=="model.language_model.embed_tokens.weight":
        rows=ne.any(1); print(f"词嵌入：{rows.sum().item()}/{rows.numel()} 行有改动；元素改动 {n/p[k].numel():.2%}")
        e=g["语言模型(不含词嵌入)"]
    elif part=="语言模型":
        e=g["语言模型(不含词嵌入)"]; e[0]+=n; e[1]+=p[k].numel(); e[2]+=(p[k].double()-b[k].double()).pow(2).sum().item(); e[3]+=b[k].double().pow(2).sum().item()
for part,(n,t,dd,bb) in g.items():
    print(f"{part}：{n:,}/{t:,} 个元素不同 = {n/t:.2%}；整体相对幅度 ‖Δ‖/‖W‖ = {(dd/bb)**0.5 if bb else 0:.4f}")
lm=[s for s in same if s[0].startswith("model.language_model.")]
print(f"完全没变的张量 {len(same)} 个，其中语言模型 {len(lm)} 个：")
for k,s in lm: print("  ",k.replace("model.language_model.",""),s)
