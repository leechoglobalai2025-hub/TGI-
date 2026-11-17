# 事后分析（2026-11-16，A 结果出来之后写的，不是预先写定的检验）：
# 各层幽灵方向 v_l 本身有多少能量落在词表投影的近零空间（奇异值最小的 51 个方向）。随机方向期望 0.05。
import os
import numpy as np, torch
from safetensors.torch import load_file
P=os.path.join(os.environ.get("TGI_MODEL_DIR", "models/Qwen3.5-0.8B"), "model.safetensors-00001-of-00001.safetensors")
sd=load_file(P); LM="model.language_model."
E=sd[LM+"embed_tokens.weight"].double()*sd[LM+"norm.weight"].double()
ev,U=torch.linalg.eigh(E.T@E); V0=U[:,:51]
z=np.load("data/rcs_mit_Qwen3.5-0.8B_v2/collect.npz")
g=torch.tensor(z["rep_h"]-z["new_h"]).double()
rho=((g@V0)**2).sum(1)/(g**2).sum(1)
print("各层幽灵方向的 ρ：", " ".join(f"L{l}:{r:.3f}" for l,r in enumerate(rho.tolist())))
for l in (15,19,23):
    W=sd[f"{LM}layers.{l}.mlp.down_proj.weight"].double()
    r=((V0.T@W)**2).sum(0)/(W**2).sum(0)
    print(f"L{l} 全层神经元 ρ：中位 {r.median():.3f}，95 分位 {r.quantile(0.95):.3f}，ρ>0.5 的 {int((r>0.5).sum())} 个")
