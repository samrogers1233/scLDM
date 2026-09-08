"""Cell-type perturbation dataset loader ."""
from distutils.spawn import spawn
import random
import blobfile as bf
import numpy as np
import torch as th
import os
import pickle
import torch as th
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import yaml

import muon as mu
from muon import MuData
from scipy.sparse import issparse
from ..vae.models.base.vae_model import EncoderModel
import ot


def load_data_cell(
    *,
    batch_size,
    data_dir,
    ae_path=None,
    ctrl_dim=0,
    pert_dim=0,
    deterministic=False,
    random_flip=True,
    num_workers=0,
    frame_gap=1,
    drop_last=True,
    condition='cell_type',
    encoder_config='default',
    dev="cuda:0",
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    print(f"load multi-omi data from {data_dir}......")
    dataset = MultimodalDataset_cell(
        data_path=data_dir,
        ae_path=ae_path,
        condition=condition,
        encoder_config=encoder_config,
        dev=dev,
        condition_key="condition",
        control_value="control",
        perturbed_value="stimulated",
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last
        )

    while True:
        yield from loader


class MultimodalDataset_cell(Dataset):
    def __init__(
        self,
        data_path,
        ae_path=None,
        condition='cell_label',
        encoder_config='default',
        dev="cuda:0",
        condition_key="condition",
        control_value="Control",
        perturbed_value="Hpoly.Day10",
    ):
        super().__init__()
        self.data_path = data_path

        self.condition = condition
        self.condition_key = condition_key
        self.control_value = control_value
        self.perturbed_value = perturbed_value
        self.adata = sc.read(data_path)
        adata_all = self.adata.copy()
        adata_ctrl = self.adata[self.adata.obs[condition_key] == control_value].copy()
        adata_pert = self.adata[self.adata.obs[condition_key] == perturbed_value].copy()
        celltype_num = np.unique(adata_pert.obs[condition]).shape[0]

        labels = adata_pert.obs[condition].values
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        self.classes = label_encoder.transform(labels)

        print("loading encoder and processing data...")
        self.adata_ctrl, self.adata_pert, self.ctrl_std10, self.pert_std10 = self.encode_raw_data(
            ae_path, adata_all, adata_ctrl, adata_pert, celltype_num, encoder_config, dev
        )
        np.savez('/'.join(ae_path.split('/')[:-2]) + '/norm_factor18.npz',
                 ctrl_std=float(self.ctrl_std10), pert_std=float(self.pert_std10))
        print("done!")

    def encode_raw_data(self, ae_path, adata_all, adata_ctrl, adata_pert, celltype_num, encoder_config, dev):
        with open(encoder_config, 'r') as file:
            yaml_content = file.read()
        autoencoder_args = yaml.safe_load(yaml_content)

        encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
                                     n_cat=celltype_num,
                                     conditioning_covariate=self.condition,
                                     encoder_type='learnt_autoencoder',
                                     **autoencoder_args)

        encoder_model.load_state_dict(torch.load(ae_path, map_location=torch.device(dev))["state_dict"])
        encoder_model.eval()

        # rescaling into std = 1
        ctrl, pert = self.latent_predict(adata_all, encoder_model)
        ctrl_std10 = ctrl.std(0).mean() * 10
        pert_std10 = pert.std(0).mean() * 10

        ctrl_np = np.expand_dims(ctrl / ctrl_std10, 1)  # [N, 1, D]
        pert_np = np.expand_dims(pert / pert_std10, 1)
        return ctrl_np, pert_np, ctrl_std10, pert_std10

    def latent_predict(self, adata_all, encoder_model):
        ctrl_adata = adata_all[adata_all.obs[self.condition_key] == self.control_value].copy()
        pert_adata = adata_all[adata_all.obs[self.condition_key] == self.perturbed_value].copy()

        z_ctrl = self.encode_data(ctrl_adata, encoder_model)
        z_pert = self.encode_data(pert_adata, encoder_model)

        # optimal-transport matching between ctrl and pert latents
        M = ot.dist(z_pert, z_ctrl, metric='euclidean')
        G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
                   torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
                   torch.tensor(M), numItermax=100000)
        match_idx = torch.max(G, 1)[1].numpy()
        ctrl_new = z_ctrl[match_idx]

        return ctrl_new, z_pert

    def encode_data(self, adata, encoder_model):
        ctrl_X = adata.X

        if issparse(ctrl_X):
            ctrl_X = ctrl_X.toarray()

        ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)

        from scLDM.perturbation.vae.data.utils import normalize_expression
        ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')

        with torch.no_grad():
            z_ctrl, _, _ = encoder_model.encode({"X_norm": ctrl_X_norm})

        return z_ctrl.detach().cpu().numpy()

    def __len__(self):
        return self.adata_ctrl.shape[0]

    def get_item(self, idx):
        pert = self.adata_pert[idx]
        pert_ctrl_cond = self.adata_ctrl[idx]
        return pert, self.classes[idx], pert_ctrl_cond

    def __getitem__(self, idx):
        audio, class_num, pert_ctrl = self.get_item(idx)
        return audio, class_num, pert_ctrl
