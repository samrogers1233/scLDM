import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import scduo.scduo_perturbation.configs as configs
from script.training_diffusion.train import *
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch
from scduo.scduo_perturbation.configs import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scduo.scduo_perturbation.diffusion.diffusion import *
from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
from collections import defaultdict
matplotlib.use('Agg')

device = configs.DEVICE
pin = (torch.device(device).type == "cuda")

train_dataset_path = "/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad"
condition_key = "condition"
control_value="control"
perturbed_value="stimulated"
condition = 'cell_type'
train_data= sc.read(train_dataset_path)


ctrl_train_loader = train_data[train_data.obs[condition_key] == control_value].copy()
pert_train_loader = train_data[train_data.obs[condition_key] == perturbed_value].copy()


encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
dataset_path = '/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad'
covariate_keys = "cell_type" 
num_class = 6
ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v1.ckpt"
adata = sc.read(dataset_path)

with open(encoder_config, 'r') as file:
        yaml_content = file.read()
autoencoder_args = yaml.safe_load(yaml_content)
encoder_model = EncoderModel(in_dim=adata.shape[1],
                                    n_cat=num_class,
                                    conditioning_covariate="cell_type", 
                                    encoder_type='learnt_autoencoder',
                                    **autoencoder_args)

encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
encoder_model = encoder_model.to(device).eval()


def get_norm_data(adata_ctrl):
    batch = {}
    ctrl_X = adata_ctrl.X
    if issparse(ctrl_X):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=device)
    from scduo.scduo_perturbation.vae.data.utils import normalize_expression
    ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
    batch["X_norm"] = torch.tensor(ctrl_X_norm, dtype=encoder_model.dtype)
    return batch


ctrl_train_data = get_norm_data(ctrl_train_loader)["X_norm"]
pert_train_data = get_norm_data(pert_train_loader)["X_norm"]

data = pert_train_data

z_list = []
with torch.no_grad():
        dataloader = DataLoader(data, batch_size=100, shuffle=False, drop_last=False)
        for idx, batch in enumerate(dataloader):
            batch = batch.to(configs.DEVICE)
            z,_,_ = encoder_model.encode({"X_norm": batch})
            z_list.append(z)
z_list = torch.cat(z_list, dim=0)


labels = ctrl_train_loader.obs[condition].values
label_encoder = LabelEncoder()
label_encoder.fit(labels)
classes= label_encoder.transform(labels)
classes = torch.tensor(classes, dtype=torch.long, device=configs.DEVICE)
num_classes = ctrl_train_loader.obs[condition].nunique()

dz = z_list.shape[1]
centroids = torch.zeros(num_classes, dz, device=device)
counts    = torch.zeros(num_classes, dtype=torch.long, device=device)

for c in range(num_classes):
    mask = (classes == c)
    if mask.any():
        centroids[c] = z_list[mask].mean(dim=0).cpu()
        counts[c]    = mask.sum()


class ProtoEmbedder(nn.Module):
    def __init__(self, dz, de=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, 2*de),
            nn.SiLU(),
            nn.Linear(2*de, de),
        )

    def forward(self, mu):                  # mu: [C, dz]
        e = self.net(mu)                    # [C, de]
        e = F.normalize(e, dim=-1)          # 单位化，便于余弦度量
        return e


def separation_loss(E, margin=0.3):
    # E: [C, de]（已单位化）
    S = E @ E.t()                           # [C, C] 余弦相似度
    C = E.size(0)
    mask = ~torch.eye(C, dtype=torch.bool, device=E.device)
    cos_ij = S[mask]
    # 约束：cos(i,j) <= 1 - margin  -> ReLU(cos - (1 - margin))
    return F.relu(cos_ij - (1.0 - margin)).mean()



# 训练嵌入器
de = 128          # 你要的条件维度
margin = 0.1
steps  = 1500
lr     = 1e-3

embedder = ProtoEmbedder(dz, de).to(device)
opt = torch.optim.AdamW(embedder.parameters(), lr=lr, weight_decay=0.0)

embedder.train()
for t in range(steps):
    E = embedder(centroids)              # [C, de], 已单位化
    loss_sep = separation_loss(E, margin=margin)
    loss = loss_sep                      # 也可以加点 L2 正则 or 稍许平滑项

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # 可选打印
    if t%50==0:
        print(f"[{t+1}/{steps}] sep_loss={loss_sep.item():.4f}")

embedder.eval()
with torch.no_grad():
    E = embedder(centroids)              # [C, de], 单位化好的最终嵌入

# ====== 4) 保存结果（矩阵 + 名称映射），以及可选初始化到 nn.Embedding ======
# 映射：类 id -> 细胞类型名
id2name = {i: n for i, n in enumerate(label_encoder.classes_.tolist())}

np.savez("/home/wuboyang/scduo-new/script/training_diffusion/outputs/embedding/celltype_emb.npz",
         emb=E.detach().cpu().numpy(),
         id2name=np.array([id2name[i] for i in range(num_classes)], dtype=object))

# 如果你希望后续用 nn.Embedding 直接拿到这个结果：
label_emb = nn.Embedding(num_classes, de).to(device)
with torch.no_grad():
    label_emb.weight.copy_(E)            # 把学到的嵌入拷进来
label_emb.weight.requires_grad_(True)    # 后续可继续微调；若不想训练则设 False


print(E[:, :10])  # 打印前 10 个嵌入向量