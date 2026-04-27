import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scanpy as sc
import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli
import muon as mu
import yaml
import seaborn as sns
from tqdm import tqdm
from scLDM.perturbation.vae.data.data_loader import RNAseqLoader
from scLDM.perturbation.vae.models.base.vae_model import EncoderModel
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


latent_data = np.load("path/to/latent.npz")
z_ctrl = torch.tensor(latent_data["z_ctrl"]).cuda()
z_pert = torch.tensor(latent_data["z_pert"]).cuda()


encoder_config = "path/to/encoder/default.yaml"
dataset_path = "path/to/dataset.h5ad"
covariate_keys = "celltype"
num_class = 22
ae_path = "path/to/autoencoder.ckpt"


adata = sc.read(dataset_path)


dataset = RNAseqLoader(
    data_path=dataset_path,
    layer_key='X_counts',
    covariate_keys=[covariate_keys],
    subsample_frac=1,
    encoder_type='learnt_autoencoder',
    condition_key="condition",
    control_value="control",
    perturbed_value="severe COVID-19",
)
gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}


X_ctrl_true = dataset.X["ctrl"].cuda()
X_pert_true = dataset.X["pert"].cuda()


with open(encoder_config, 'r') as file:
    yaml_content = file.read()
autoencoder_args = yaml.safe_load(yaml_content)

autoencoder_args['encoder_kwargs']['ctrl']['norm_type'] = 'batchnorm'
autoencoder_args['encoder_kwargs']['pert']['norm_type'] = 'batchnorm'
encoder_model = EncoderModel(
    in_dim=gene_dim,
    n_cat=adata.obs[covariate_keys].unique().shape[0],
    conditioning_covariate=covariate_keys,
    encoder_type='proportions',
    **autoencoder_args,
)

encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])


encoder_model.eval()
encoder_model.cuda()
with torch.no_grad():
    sf_ctrl = X_ctrl_true.sum(1, keepdim=True)
    sf_pert = X_pert_true.sum(1, keepdim=True)

    X_ctrl_recon = encoder_model.decode_one(z_ctrl, mod="ctrl")
    X_pert_recon = encoder_model.decode_one(z_pert, mod="pert")

Xc_true, Xc_pred = X_ctrl_true.cpu().numpy(), X_ctrl_recon.cpu().numpy()
Xp_true, Xp_pred = X_pert_true.cpu().numpy(), X_pert_recon.cpu().numpy()


def compute_metrics(X_true, X_pred):
    pearson_list, r2_list = [], []
    for i in range(X_true.shape[0]):
        true_row, pred_row = X_true[i], X_pred[i]
        if np.std(true_row) == 0:  # skip constant rows to avoid pearson NaN
            continue
        pearson, _ = pearsonr(true_row, pred_row)
        r2 = r2_score(true_row, pred_row)
        pearson_list.append(pearson)
        r2_list.append(r2)
    return pearson_list, r2_list


pearson_ctrl, r2_ctrl = compute_metrics(Xc_true, Xc_pred)
pearson_pert, r2_pert = compute_metrics(Xp_true, Xp_pred)

print("Control group:")
print(f"  Pearson (mean ± std): {np.mean(pearson_ctrl):.4f} ± {np.std(pearson_ctrl):.4f}")
print(f"  R² (mean ± std):      {np.mean(r2_ctrl):.4f} ± {np.std(r2_ctrl):.4f}")

print("Perturbed group:")
print(f"  Pearson (mean ± std): {np.mean(pearson_pert):.4f} ± {np.std(pearson_pert):.4f}")
print(f"  R² (mean ± std):      {np.mean(r2_pert):.4f} ± {np.std(r2_pert):.4f}")
