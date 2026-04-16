# import torch
# from torch.utils.data import DataLoader
# import scanpy as sc
# from torch.distributions import kl_divergence as kl
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import ot   
# from scipy.sparse import issparse
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import numpy as np
# import yaml
# #from scdiffusionX.utils import *
# from scduo.scduo_perturbation.vae.data.data_loader import RNAseqLoader
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.stats import pearsonr
# from sklearn.metrics import r2_score


# encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
# dataset_path = '/home/wuboyang/scduo-new/dataset/gene_perturb_data/data_processed.h5ad'
# covariate_keys = "celltype" 
# num_class = 7
# ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v3.ckpt"
# adata_all = sc.read(dataset_path)

# with open(encoder_config, 'r') as file:
#     autoencoder_args = yaml.safe_load(file)


# encoder_model = EncoderModel(
#     in_dim=adata_all.shape[1],
#     n_cat=adata_all.obs[covariate_keys].unique().shape[0],
#     conditioning_covariate=covariate_keys,
#     encoder_type='learnt_autoencoder',
#     **autoencoder_args
# )
# encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
# encoder_model.eval()

# adata_nk = adata_all[adata_all.obs['cell_type'] == 'NK'].copy()
# adata = adata_all[adata_all.obs['cell_type'] != 'NK'].copy()

# def encode_data(adata,encoder_model):
#     adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
#     adata_pert = adata[adata.obs["condition"] == "stimulated"].copy()

#     ctrl_X = adata_ctrl.X
#     pert_X = adata_pert.X

#     if issparse(ctrl_X):
#         ctrl_X = ctrl_X.toarray()
#     if issparse(pert_X):
#         pert_X = pert_X.toarray()

#     ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#     pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#     from scduo.scduo_perturbation.vae.data.utils import normalize_expression
#     ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#     pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
#     with torch.no_grad():
#         z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
#         z_pert,_,_ = encoder_model.encode({"X_norm":pert_X_norm})
#     # latent_adata = sc.AnnData(X=z["ctrl"], obs=adata_ctrl.obs.copy())
#     return z_ctrl.detach().cpu().numpy(), z_pert.detach().cpu().numpy(), ctrl_X_norm, pert_X_norm

# nk_ctrl, nk_pert, nk_ctrl_norm, nk_pert_norm = encode_data(adata_nk, encoder_model)
# z_ctrl, z_pert,_,_ = encode_data(adata, encoder_model)


# M = ot.dist(z_pert, z_ctrl, metric='euclidean')
# G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
#             torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
#             torch.tensor(M), numItermax=100000)
# match_idx = torch.max(G, 0)[1].numpy()
# stim_new = z_pert[match_idx]
# delta_list = stim_new - z_ctrl

# cos_sim = cosine_similarity(np.array(nk_ctrl),
#                             np.array(z_ctrl))
# ratio=0.05
# n_top = int(np.ceil(z_ctrl.shape[0] * ratio))
# delta_pred_list = []
# for i in range(cos_sim.shape[0]):
#     top_indices = np.argsort(cos_sim)[i][-n_top:]
#     normalized_weights = cos_sim[i][top_indices] / np.sum(cos_sim[i][top_indices])
#     delta_pred = np.sum(normalized_weights[:, np.newaxis] *
#                     np.array(delta_list)[top_indices], axis=0)
#     delta_pred_list.append(delta_pred)
# delta_pred_ot = np.stack(delta_pred_list)
# nk_pred = nk_ctrl + delta_pred_ot


# # z_true = nk_pert.mean(axis=0)
# # z_pred = nk_pred.mean(axis=0)
# x_hat = encoder_model.decode(torch.tensor(nk_pred, dtype=encoder_model.dtype, device=encoder_model.device))
# z_true = nk_pert_norm.mean(axis=0)
# z_pred = x_hat.mean(axis=0)

# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return x  # already numpy

# z_pred = to_numpy(z_pred)
# z_true = to_numpy(z_true)
# mse = mean_squared_error(z_true, z_pred)
# mae = mean_absolute_error(z_true, z_pred)
# r2 = r2_score(z_true, z_pred)
# corr, _ = pearsonr(z_true, z_pred)
# print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson R: {corr:.4f}")









import torch
from torch.utils.data import DataLoader
import scanpy as sc
from torch.distributions import kl_divergence as kl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ot   
from scipy.sparse import issparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import yaml
#from scdiffusionX.utils import *
from scduo.scduo_perturbation.vae.data.data_loader import RNAseqLoader
from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
dataset_path = '/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad'
covariate_keys = "cell_type" 
num_class = 7
ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v2.ckpt"
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


adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
adata_pert = adata[adata.obs["condition"] == "stimulated"].copy()

ctrl_X = adata_ctrl.X
pert_X = adata_pert.X

if issparse(ctrl_X):
    ctrl_X = ctrl_X.toarray()
if issparse(pert_X):
    pert_X = pert_X.toarray()

ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
from scduo.scduo_perturbation.vae.data.utils import normalize_expression
ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
with torch.no_grad():
    z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
    z_pert,_,_ = encoder_model.encode({"X_norm":pert_X_norm})
delta = z_pert.mean()- z_ctrl.mean()
pred = z_ctrl + delta


# M = ot.dist(z_pert, z_ctrl, metric='euclidean')
# G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
#             torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
#             torch.tensor(M), numItermax=100000)
# match_idx = torch.max(G, 0)[1].numpy()
# stim_new = z_pert[match_idx]
# delta_list = stim_new - z_ctrl

# cos_sim = cosine_similarity(np.array(nk_ctrl),
#                             np.array(z_ctrl))
# ratio=0.05
# n_top = int(np.ceil(z_ctrl.shape[0] * ratio))
# delta_pred_list = []
# for i in range(cos_sim.shape[0]):
#     top_indices = np.argsort(cos_sim)[i][-n_top:]
#     normalized_weights = cos_sim[i][top_indices] / np.sum(cos_sim[i][top_indices])
#     delta_pred = np.sum(normalized_weights[:, np.newaxis] *
#                     np.array(delta_list)[top_indices], axis=0)
#     delta_pred_list.append(delta_pred)
# delta_pred_ot = np.stack(delta_pred_list)
# nk_pred = nk_ctrl + delta_pred_ot


# z_true = nk_pert.mean(axis=0)
# z_pred = nk_pred.mean(axis=0)

x_hat = encoder_model.decode(torch.tensor(pred, dtype=encoder_model.dtype, device=encoder_model.device))
pred_mean = x_hat.mean(axis=0)
pert_mean = pert_X_norm.mean(axis=0)
ctrl_mean = ctrl_X_norm.mean(axis=0)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x  # already numpy

pred_mean = to_numpy(pred_mean)
pert_mean= to_numpy(pert_mean)
mse = mean_squared_error(pred_mean,pert_mean)
mae = mean_absolute_error(pred_mean,pert_mean)
r2 = r2_score(pred_mean,pert_mean)
corr, _ = pearsonr(pred_mean,pert_mean)
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson R: {corr:.4f}")




#scgen基因扰动预测结果MSE: 0.0007, MAE: 0.0178, R²: 0.9352, Pearson R: 0.9760
