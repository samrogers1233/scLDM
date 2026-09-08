import math
import os
import random

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import yaml
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam

import scLDM.perturbation.configs as configs
from scLDM.perturbation.diffusion.AttnUnet import Unet
from scLDM.perturbation.diffusion.PlainUnet import SimpleUnet_plain
from scLDM.perturbation.diffusion.diffusion import train_translators
from scLDM.perturbation.vae.data.utils import normalize_expression
from scLDM.perturbation.vae.models.base.vae_model import EncoderModel


os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = configs.DEVICE


# ---------- edit these ----------
TRAIN_DATASET_PATH = "path/to/train.h5ad"
VAL_DATASET_PATH = "path/to/valid.h5ad"
ENCODER_CONFIG = "path/to/encoder/default.yaml"
AE_PATH = "path/to/autoencoder.ckpt"
CELLTYPE_EMB_PATH = "path/to/celltype_emb.npz"
OUTPUT_DIR = "path/to/diffusion_outputs"

CONDITION = "cell_type"
CONDITION_KEY = "condition"
CONTROL_VALUE = "control"
PERTURBED_VALUE = "stimulated"
EMBEDDING_DIM = 128
NUM_CLASSES_VAE = 6  # n_cat used when the VAE was trained


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(plain=False):
    # channels=3 matches train_translators' cat([x_t, xB0, emb], dim=1) input.
    if not plain:
        model = Unet(
            configs.DIM,
            channels=3,
            out_dim=1,
            dim_mults=(1, 2, 4, 8, 8),
        )
    else:
        model = SimpleUnet_plain(in_dim=1, dim=64, out_dim=1)
    print("Num params:", sum(p.numel() for p in model.parameters()))
    return model


@torch.no_grad()
def encode_raw_data(encoder_model, adata, bs=100):
    """Encode an AnnData through the VAE and rescale to std=0.1 in latent space."""
    zs = []
    num_batches = math.ceil(adata.shape[0] / bs)
    for i in range(num_batches):
        x = adata[i * bs:(i + 1) * bs].X
        if issparse(x):
            x = x.toarray()
        x = torch.tensor(x, dtype=encoder_model.dtype, device=encoder_model.device)
        x_norm = normalize_expression(x, x.sum(), encoder_type='learnt_autoencoder')
        z, _, _ = encoder_model.encode({"X_norm": x_norm})
        zs.append(z)
    z_all = torch.cat(zs)
    z_std10 = z_all.std(0).mean() * 10
    return z_all / z_std10


def train_with_early_stopping_single(
    gen, optimizer_gen,
    encoder_model,
    ctrl_train, pert_train,
    num_epochs=150, patience=50, iterations_per_epoch=100,
    classes=None, label_emb=None,
    save_dir=OUTPUT_DIR,
):
    """Train the latent-space translator (single-model variant).

    Validation loss is not wired in: `evaluate_translators` still expects the
    2-channel input pattern, while `train_translators` uses 3 channels
    (xAtA, xB0, label emb). Early stopping therefore monitors train loss.
    """
    z_ctrl_train = encode_raw_data(encoder_model, ctrl_train)
    z_pert_train = encode_raw_data(encoder_model, pert_train)

    best_loss = float('inf')
    wait = 0
    train_losses = []

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_gen2.pth')

    print('training translator (single-model) ......')
    for ep in range(1, num_epochs + 1):
        avg_train = train_translators(
            ep, gen, optimizer_gen, z_pert_train, z_ctrl_train, classes, label_emb,
            iterations=iterations_per_epoch, batch_size=configs.BATCH_SIZE,
            device=device, T=configs.TIMESTEPS,
        )
        train_losses.append(avg_train)
        print(f"Epoch {ep}: Train {avg_train:.4f}")

        if avg_train < best_loss:
            best_loss = avg_train
            wait = 0
            torch.save(gen.state_dict(), best_path)
            print("improved, saved:", best_path)
        else:
            wait += 1
            print(f"No improvement for {wait} epochs")
            if wait >= patience:
                print("Early stopping.")
                break

    gen.load_state_dict(torch.load(best_path, map_location=device))
    return gen, train_losses


if __name__ == "__main__":
    setup_seed(19193)

    train_data = sc.read(TRAIN_DATASET_PATH)
    ctrl_train = train_data[train_data.obs[CONDITION_KEY] == CONTROL_VALUE].copy()
    pert_train = train_data[train_data.obs[CONDITION_KEY] == PERTURBED_VALUE].copy()

    labels = ctrl_train.obs[CONDITION].values
    classes_np = LabelEncoder().fit_transform(labels)
    classes = torch.tensor(classes_np, dtype=torch.long, device=device)
    num_classes = ctrl_train.obs[CONDITION].nunique()

    label_emb = nn.Embedding(num_classes, EMBEDDING_DIM).to(device)
    E = np.load(CELLTYPE_EMB_PATH)["emb"]
    E = torch.as_tensor(E, dtype=torch.float32, device=device)
    with torch.no_grad():
        label_emb.weight.copy_(E)
        label_emb.weight.requires_grad_(False)

    with open(ENCODER_CONFIG, 'r') as f:
        autoencoder_args = yaml.safe_load(f)
    encoder_model = EncoderModel(
        in_dim=train_data.shape[1],
        n_cat=NUM_CLASSES_VAE,
        conditioning_covariate=CONDITION,
        encoder_type='learnt_autoencoder',
        **autoencoder_args,
    )
    encoder_model.load_state_dict(torch.load(AE_PATH, map_location=device)["state_dict"])
    encoder_model.eval()
    encoder_model.to(device)

    gen = get_model().to(device)
    optim_gen = Adam(gen.parameters(), lr=configs.GENA_LR)

    best_gen, train_losses = train_with_early_stopping_single(
        gen, optim_gen,
        encoder_model,
        ctrl_train, pert_train,
        num_epochs=configs.EPOCHS,
        patience=50,
        iterations_per_epoch=100,
        classes=classes,
        label_emb=label_emb,
    )
