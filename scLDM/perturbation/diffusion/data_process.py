import numpy as np
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
import yaml
from scipy.sparse import issparse
from ..vae.models.base.vae_model import EncoderModel
from sklearn.metrics.pairwise import cosine_similarity
import ot 


def load_gene2vec_txt_simple(path, dtype=np.float32):
    with open(path, "r") as f:
        num_genes, dim = map(int, f.readline().strip().split())
        genes = []
        embs = np.zeros((num_genes, dim), dtype=dtype)
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            genes.append(parts[0])
            embs[i] = np.asarray(parts[1:], dtype=dtype)
    gene2idx = {g: i for i, g in enumerate(genes)}
    return genes, embs, gene2idx


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
    gene2vec_path="path/to/gene2vec.txt",
    perturbation_key="condition",
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    print(f"load multi-omi data from {data_dir}......")
    dataset = MultimodalDataset_cell(
        data_path = data_dir,
        ae_path=ae_path,
        condition=condition,
        encoder_config=encoder_config,
        dev=dev,
        condition_key="cp_type",
        control_value="control",
        perturbed_value="stimulated",
        gene2vec_path=gene2vec_path,
        perturbation_key=perturbation_key,
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=drop_last
        )
        
    while True:
        yield from loader



class MultimodalDataset_cell(Dataset):
    def __init__(
        self,
        data_path,
        ae_path=None,
        condition='cell_type',
        encoder_config='default',
        dev="cuda:0",
        condition_key="cp_type",
        control_value="control",
        perturbed_value="stimulated",
        gene2vec_path="path/to/gene2vec.txt",
        perturbation_key="condition",
    ):
        super().__init__()
        self.data_path = data_path

        self.condition = condition
        self.condition_key = condition_key
        self.adata = sc.read(data_path)
        adata_all = self.adata.copy()
        adata_ctrl = self.adata[self.adata.obs[condition_key] == control_value].copy()
        adata_pert = self.adata[self.adata.obs[condition_key] == perturbed_value].copy()
        celltype_num = np.unique(adata_pert.obs[condition]).shape[0]
        self.perturbation_key = perturbation_key
        self.gene2vec_path = gene2vec_path

        labels = adata_pert.obs[condition].values
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        self.classes = label_encoder.transform(labels)

        if self.gene2vec_path is not None:
            self.gene_emb, self_gene_emb_dim = self._build_perturbation_embeddings(adata_pert)
        assert self.gene_emb.shape[0] == adata_pert.shape[0]

        print("loading encoder and processing data...")
        self.adata_ctrl, self.adata_pert, self.ctrl_std10, self.pert_std10 = self.encode_raw_data(ae_path, adata_all, adata_ctrl, adata_pert,celltype_num,encoder_config,dev)
        np.savez('/'.join(ae_path.split('/')[:-2])+'/norm_factor19.npz',ctrl_std=float(self.ctrl_std10),pert_std=float(self.pert_std10))
        print("done!")

    def _build_perturbation_embeddings(self, adata_pert):
        genes, embs, gene2idx = load_gene2vec_txt_simple(self.gene2vec_path, dtype=np.float32)
        dim = embs.shape[1]
        pert_series = adata_pert.obs[self.perturbation_key].astype(str)
        nperts_value = adata_pert.obs["nperts"].astype(int)

        vectors = []
        missing = []
        for g, n in zip(pert_series, nperts_value):
            if n == 1:
                if g in gene2idx:
                    vectors.append(embs[gene2idx[g]])
                else:
                    vectors.append(np.zeros(dim, dtype=np.float32))
                    missing.append(g)
            elif n == 2:
                g1, g2 = g.split("_")
                vec = np.zeros(dim, dtype=np.float32)
                if g1 in gene2idx:
                    vec += embs[gene2idx[g1]]
                else:
                    missing.append(g1)
                if g2 in gene2idx:
                    vec += embs[gene2idx[g2]]
                else:
                    missing.append(g2)
                vectors.append(vec / 2)
            else:
                print(f"[gene2vec] Warning: 发现 nperts > 2 的扰动 {g}")
                
        pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
        pert_emb_dim = dim
        uniq_missing = sorted(set(missing))
        if uniq_missing:
            print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
        print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
        return pert_emb, pert_emb_dim

    def encode_raw_data(self, ae_path, adata_all, adata_ctrl, adata_pert,celltype_num,encoder_config,dev):
        with open(encoder_config, 'r') as file:
            yaml_content = file.read()
        autoencoder_args = yaml.safe_load(yaml_content)

        # Initialize encoder
        encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
                                            n_cat=celltype_num,
                                            conditioning_covariate=self.condition, 
                                            encoder_type='learnt_autoencoder',
                                            **autoencoder_args)
        
        # Load weights 
        encoder_model.load_state_dict(torch.load(ae_path, map_location=torch.device(dev))["state_dict"])
        encoder_model.eval()
        
        # rescaling into std = 1
        ctrl, pert = self.latent_predict(adata_all, encoder_model)
        ctrl_std10 = ctrl.std(0).mean()*10
        pert_std10 = pert.std(0).mean()*10

        ctrl_np = np.expand_dims(ctrl / ctrl_std10, 1)  # [N, 1, D]
        pert_np = np.expand_dims(pert / pert_std10, 1)
        return ctrl_np, pert_np, ctrl_std10, pert_std10
        
    def latent_predict(self, adata_all, encoder_model):
        ctrl_adata = adata_all[adata_all.obs['cp_type'] == 'control'].copy()
        pert_adata = adata_all[adata_all.obs['cp_type'] == 'stimulated'].copy()

        z_ctrl = self.encode_data(ctrl_adata, encoder_model)
        z_pert = self.encode_data(pert_adata, encoder_model)


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
                z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
            
            return z_ctrl.detach().cpu().numpy()

    def __len__(self):
        return self.adata_pert.shape[0]

    def get_item(self, idx):
   
        pert = self.adata_pert[idx]
        pert_ctrl_cond = self.adata_ctrl[idx]
        pert_gene_cond = self.gene_emb[idx] 

        return pert, self.classes[idx], pert_ctrl_cond, pert_gene_cond
    
    def __getitem__(self, idx):
        audio, class_num, pert_ctrl, pert_gene= self.get_item(idx)

        return audio, class_num, pert_ctrl, pert_gene