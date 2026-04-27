import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import scanpy as sc
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from scLDM.perturbation.vae.data.data_loader import RNAseqLoader
from scLDM.perturbation.vae.models.base.vae_model import EncoderModel


encoder_config = "path/to/encoder/default.yaml"
dataset_path = "path/to/dataset.h5ad"
covariate_keys = "celltype"
ae_path = "path/to/autoencoder.ckpt"

adata = sc.read(dataset_path)

dataset = RNAseqLoader(
    data_path=dataset_path,
    layer_key='X_counts',
    covariate_keys=[covariate_keys],
    subsample_frac=1,
    encoder_type='learnt_autoencoder',
    condition_key="condition",
    control_value="Control",
    perturbed_value="Hpoly.Day10",
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

with open(encoder_config, 'r') as file:
    autoencoder_args = yaml.safe_load(file)

autoencoder_args['encoder_kwargs']['norm_type'] = 'batchnorm'

gene_dim = dataset.X.shape[1]
encoder_model = EncoderModel(
    in_dim=gene_dim,
    n_cat=adata.obs[covariate_keys].unique().shape[0],
    conditioning_covariate=covariate_keys,
    encoder_type='learnt_autoencoder',
    **autoencoder_args,
)

encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
encoder_model.eval()

all_x_hat = []
all_input = []
latent_list = []

with torch.no_grad():
    for batch in dataloader:
        z, mu, logvar = encoder_model.encode(batch)
        x_hat = encoder_model.decode(z)

        latent_list.append(z.cpu())
        all_x_hat.append(x_hat.cpu())
        all_input.append(batch["X_norm"].cpu())

Y_pred = torch.cat(all_x_hat, dim=0).numpy()
Y_true = torch.cat(all_input, dim=0).numpy()
latent = torch.cat(latent_list, dim=0).numpy()

from sklearn.metrics import r2_score


def evaluate_reconstruction(all_input, all_mu_hat):
    y_true = all_input.mean(axis=0)
    y_pred = all_mu_hat.mean(axis=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson R: {corr:.4f}")


evaluate_reconstruction(Y_true, Y_pred)
