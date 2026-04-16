import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端

import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.optim as optim
import muon as mu
import yaml
import os
#from scdiffusionX.utils import *

from scduo.scduo_perturbation.vae.data.data_loader import RNAseqLoader
from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel

import argparse
from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from scduo.scduo_perturbation.diffusion import dist_util
import scanpy as sc
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli
from scipy.stats import spearmanr


encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
dataset_path = '/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad'
covariate_keys = "cell_type" 
num_class = 7
ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v2.ckpt"
condition_key = "condition"
control_value="control"
perturbed_value="stimulated"  
condition= 'cell_type'  
adata = sc.read(dataset_path)

with open(encoder_config, 'r') as file:
    autoencoder_args = yaml.safe_load(file)

encoder_model = EncoderModel(
    in_dim=adata.shape[1],
    n_cat=adata.obs[covariate_keys].unique().shape[0],
    conditioning_covariate=covariate_keys,
    encoder_type='learnt_autoencoder',
    **autoencoder_args
)

encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
encoder_model.eval()

pert_seq = np.load('/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs6/sample_outputs/sample_data_alldata.npz')['pert']
type_index = np.load('/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs6/sample_outputs/sample_data_alldata.npz')['label']
idx = np.load('/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs6/sample_outputs/sample_data_alldata.npz')['idx']

# load norm factor for encoder
npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor3.npz')
ctrl_std = npzfile['ctrl_std']
pert_std = npzfile['pert_std']
z = {'pert':torch.tensor(pert_seq*pert_std).squeeze(1)}     # open  layernorm
pert_hat = encoder_model.decode(z["pert"])


adata_ctrl = adata[adata.obs[condition_key] == control_value].copy()
adata_pert = adata[adata.obs[condition_key] == perturbed_value].copy()

batch = {}
ctrl_X = adata_ctrl.X
pert_X = adata_pert.X
from scipy.sparse import issparse
if issparse(ctrl_X):
    ctrl_X = ctrl_X.toarray()
if issparse(pert_X):
    pert_X = pert_X.toarray()

ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)

from scduo.scduo_perturbation.vae.data.utils import normalize_expression
ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')


ctrl_real = ctrl_X_norm.numpy()
pert_real = pert_X_norm.numpy()

#全数据集范围采样
pert_real_mean = pert_real.mean(axis=0)
pert_gen_mean  = pert_hat.detach().mean(dim=0).cpu().numpy().ravel()

# #用id在原数据集中找对应数据
# ctrl_match = ctrl_real[idx]
# pert_match = pert_real[idx]
# pert_gen_mean  = pert_hat.detach().mean(dim=0).cpu().numpy().ravel()
# ctrl_real_mean = ctrl_match.mean(axis=0)
# pert_real_mean = pert_match.mean(axis=0)


# #用细胞类型标签随机挑选细胞
# from sklearn.preprocessing import LabelEncoder
# labels = adata_ctrl.obs[condition].values
# label_encoder = LabelEncoder()
# label_encoder.fit(labels)
# real_labels = label_encoder.transform(labels)


# from collections import defaultdict
# celltype_to_indices = defaultdict(list)
# for idx, label in enumerate(real_labels):
#     celltype_to_indices[label].append(idx)
# real_ctrl_match = []
# real_pert_match = []
# for label in type_index:
#     idx_pool = celltype_to_indices[label]
#     sampled_idx = np.random.choice(idx_pool)
#     real_ctrl_match.append(ctrl_real[sampled_idx])
#     real_pert_match.append(pert_real[sampled_idx])
# real_ctrl_match = np.stack(real_ctrl_match)
# real_pert_match = np.stack(real_pert_match)
# real_ctrl_match = torch.as_tensor(real_ctrl_match, dtype=torch.float32)
# real_pert_match = torch.as_tensor(real_pert_match, dtype=torch.float32)
# ctrl_real_mean = real_ctrl_match.mean(axis=0).detach().cpu().numpy()
# pert_real_mean = real_pert_match.mean(axis=0).detach().cpu().numpy()
# pert_gen_mean  = pert_hat.detach().mean(dim=0).cpu().numpy().ravel()
# # pert_real_mean = real_pert_match.detach().mean(dim=0).cpu().numpy().ravel()


# delta_true = pert_real_mean - ctrl_real_mean
# delta_pred = pert_gen_mean - ctrl_real_mean


# 计算 Pearson 和 Spearman 相关系数
pearson_pert = np.corrcoef(pert_gen_mean, pert_real_mean)[0, 1]
spearman_pert,_ = spearmanr(pert_gen_mean, pert_real_mean)
print(f" - Pearson:  {pearson_pert:.4f}")
print("Spearman correlation:", spearman_pert)


#计算r2
from sklearn.metrics import r2_score
def evaluate_reconstruction(all_input, all_mu_hat):
    y_true = all_input
    y_pred = all_mu_hat
    r2 = r2_score(y_true, y_pred)
    print(f"  R²: {r2:.4f}")
evaluate_reconstruction(pert_gen_mean, pert_real_mean)


# #计算MMD
# from sklearn.metrics.pairwise import rbf_kernel
# def compute_mmd(x, y, gamma=1.0):
#     """x, y: shape (n_samples, n_features)"""
#     Kxx = rbf_kernel(x, x, gamma=gamma)
#     Kyy = rbf_kernel(y, y, gamma=gamma)
#     Kxy = rbf_kernel(x, y, gamma=gamma)
#     return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
# mmd_score = compute_mmd(pert_hat.detach().cpu().numpy(), real_pert_match.numpy())
# print("MMD score:", mmd_score)


# #计算lisi
# from sklearn.neighbors import NearestNeighbors
# def compute_lisi(X, labels, perplexity=30):
#     """X: cells × genes, labels: batch or group info"""
#     n_neighbors = min(len(X)-1, perplexity*3)
#     nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
#     dists, idxs = nn.kneighbors(X)
#     # 计算相似度
#     sim = np.exp(-dists ** 2 / np.mean(dists[:,1:])**2)
#     sim = sim / sim.sum(axis=1, keepdims=True)
#     # 计算 LISI
#     lisi_scores = []
#     for i in range(len(X)):
#         p = np.bincount(labels[idxs[i]], weights=sim[i], minlength=len(np.unique(labels)))
#         p = p / p.sum()
#         lisi = 1.0 / (p**2).sum()
#         lisi_scores.append(lisi)
#     return np.mean(lisi_scores)
# # 拼接真实与生成数据
X_all = np.vstack([pert_real, pert_hat.detach().cpu().numpy()])
labels = np.array([0]*len(pert_real) + [1]*len(pert_hat))  # 0=real, 1=gen
# lisi = compute_lisi(X_all, labels)
# print("LISI score:", lisi)


# # ---------- 4. QQ-plot ----------
# import statsmodels.api as sm
# sm.qqplot_2samples(pert_gen_mean, pert_real_mean, line='45')
# plt.title("QQ-plot: Generated vs Real")
# plt.show()
# plt.savefig("/home/wuboyang/scduo-new/figures/qq-plot.png", dpi=300, bbox_inches='tight')

# ---------- 5. UMAP 可视化 ----------
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
embedding = reducer.fit_transform(X_all)
plt.figure(figsize=(6,6))
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, palette={0:"blue",1:"orange"}, alpha=0.6)
plt.title("UMAP: Real vs Generated Perturbations")
plt.legend(title="Type", labels=["Real","Generated"])
plt.show()
plt.savefig("/home/wuboyang/scduo-new/figures/umap.png", dpi=300, bbox_inches='tight')

# def compute_pcc(y_true, y_pred):
#     y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
#     mean_true, mean_pred = y_true.mean(), y_pred.mean()
#     cov = np.sum((y_true - mean_true) * (y_pred - mean_pred))
#     std_true = np.sqrt(np.sum((y_true - mean_true) ** 2))
#     std_pred = np.sqrt(np.sum((y_pred - mean_pred) ** 2))
#     return cov / (std_true * std_pred + 1e-12)

# # ===== 公式实现的 R² =====
# def compute_r2(y_true, y_pred):
#     y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - y_true.mean()) ** 2)
#     return 1 - ss_res / (ss_tot + 1e-12)

# # ===== 计算指标 =====
# pcc_val = compute_pcc(delta_true, delta_pred)
# r2_val = compute_r2(delta_true, delta_pred)

# print(f"Δexpression PCC: {pcc_val:.4f}")
# print(f"Δexpression R²: {r2_val:.4f}")





#双diffusion模型的重建结果--pbmc
# - Pearson:  0.9852
# - R²: 0.9531

#单diffusion模型的重建结果--pbmc
# - Pearson:  0.9860
#   R²: 0.9551

#单diffusion模型ood的重建结果--pbcm
# - Pearson:  0.9707
#   R²: 0.9095

#单diffusion模型的重建结果--covid
#- Pearson:  0.9893
#  R²: 0.9668
# Δexpression PCC: 0.9711
# Δexpression R²: 0.9279

#单diffusion模型的重建结果--H.holy
#- Pearson:  0.9904
#  R²: 0.9744
# Δexpression PCC: 0.8749
# Δexpression R²: 0.6013
























# #单diffuison基因扰动实验
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.cluster import KMeans

# import torch
# import torch.optim as optim
# import muon as mu
# import yaml
# import os
# #from scdiffusionX.utils import *

# from scduo.scduo_perturbation.vae.data.data_loader import RNAseqLoader
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel

# import argparse
# from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )
# from scduo.scduo_perturbation.diffusion import dist_util
# import scanpy as sc
# from torch.distributions import Normal
# from scvi.distributions import NegativeBinomial
# from torch.distributions import Poisson, Bernoulli
# from scipy.stats import spearmanr


# encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
# dataset_path = '/home/wuboyang/scduo-new/dataset/gene_perturb_data/norman_ood.h5ad'
# covariate_keys = "celltype" 
# num_class = 1
# ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v8.ckpt"
# condition_key = "cp_type"
# control_value="control"
# perturbed_value="stimulated"  
# condition= 'celltype'  
# adata = sc.read(dataset_path)

# with open(encoder_config, 'r') as file:
#     autoencoder_args = yaml.safe_load(file)


# encoder_model = EncoderModel(
#     in_dim=adata.shape[1],
#     n_cat=adata.obs[covariate_keys].unique().shape[0],
#     conditioning_covariate=covariate_keys,
#     encoder_type='learnt_autoencoder',
#     **autoencoder_args
# )


# encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
# encoder_model.eval()



# pert_seq = np.load('/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs13/sample_outputs/sample_data.npz')['pert']
# type_index = np.load('/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs13/sample_outputs/sample_data.npz')['label']
# idx = np.load('/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs13/sample_outputs/sample_data.npz')['idx']

# # load norm factor for encoder
# npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor9.npz')
# ctrl_std = npzfile['ctrl_std']
# pert_std = npzfile['pert_std']
# z = {'pert':torch.tensor(pert_seq*pert_std).squeeze(1)}     # open  layernorm
# pert_hat = encoder_model.decode(z["pert"])


# adata_ctrl = adata[adata.obs[condition_key] == control_value].copy()
# adata_pert = adata[adata.obs[condition_key] == perturbed_value].copy()

# batch = {}
# ctrl_X = adata_ctrl.X
# pert_X = adata_pert.X
# from scipy.sparse import issparse
# if issparse(ctrl_X):
#     ctrl_X = ctrl_X.toarray()
# if issparse(pert_X):
#     pert_X = pert_X.toarray()

# ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
# pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)

# from scduo.scduo_perturbation.vae.data.utils import normalize_expression
# ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
# pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')


# ctrl_real = ctrl_X_norm.numpy()
# pert_real = pert_X_norm.numpy()

# pert_match = pert_real[idx]

# # 方式一：用 detach（通用，是否开启 no_grad 都安全）
# pert_gen_mean  = pert_hat.detach().mean(dim=0).cpu().numpy().ravel()
# pert_real_mean = pert_match.mean(axis=0)



# # 计算 Pearson 和 Spearman 相关系数
# # pearson_ctrl = np.corrcoef(ctrl_gen_mean, ctrl_real_mean)[0, 1]
# # spearman_ctrl = spearmanr(ctrl_gen_mean, ctrl_real_mean).correlation
# pearson_pert = np.corrcoef(pert_gen_mean, pert_real_mean)[0, 1]
# spearman_pert = spearmanr(pert_gen_mean, pert_real_mean).correlation
# print(f" - Pearson:  {pearson_pert:.4f}")



# from sklearn.metrics import r2_score
# def evaluate_reconstruction(all_input, all_mu_hat):
#     y_true = all_input
#     y_pred = all_mu_hat
#     r2 = r2_score(y_true, y_pred)

#     print(f"  R²: {r2:.4f}")

# evaluate_reconstruction(pert_gen_mean, pert_real_mean)

# #双diffusion模型的重建结果
# # - Pearson:  0.9852
# # - R²: 0.9531

# #单diffusion模型的重建结果
# # - Pearson:  0.9860
# #   R²: 0.9551

# #单diffusion模型ood的重建结果
# # - Pearson:  0.9707
# #   R²: 0.9095

# #单diffusion模型基因扰动的重建结果--norman(单扰动)
# # - Pearson:  0.9760
# #   R²: 0.9398

# #单diffusion模型基因slc4a1扰动重建--norman(单扰动)
# # - Pearson:  0.9676
# #   R²: 0.9209

# #单diffusion模型基因扰动的重建结果--adamson
# #  - Pearson:  0.9732
# #    R²: 0.9434

# #单diffusion模型基因IER3IP1扰动重建--adamson
# #  - Pearson:  0.9708
# #    R²: 0.9395

# #单diffusion模型基因扰动的重建结果--norman(单双扰动)
# #  - Pearson:  0.9776
# #    R²: 0.9424

# #单diffusion模型基因CEBPE_RUNX1T1扰动重建--norman(单双扰动)
# #  - Pearson:  0.9675
# #    R²: 0.9216

# #单diffusion模型基因扰动的重建结果--norman_ood(单双扰动)
# # - Pearson:  0.9620
# #   R²: 0.8970

