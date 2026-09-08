#scLDM
import numpy as np
import scanpy as sc
import torch
from .utils import normalize_expression, compute_size_factor_lognorm
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse import issparse

# Requires upstream preprocessing so that ctrl and pert have matching sample counts.
class RNAseqLoader:
    """Loader for paired control and perturbed scRNA-seq data from a single .h5ad file."""

    def __init__(
        self,
        data_path: str,
        layer_key: str,
        covariate_keys=None,
        subsample_frac=1,
        encoder_type="proportions",
        condition_key="condition",
        control_value="control",
        perturbed_value="perturbed",
    ):
        # Set normalization type
        self.encoder_type = encoder_type
        self.condition_key = condition_key

        # Read full AnnData object
        adata = sc.read(data_path)


        # Transform X into a tensor
        layer_data = adata.X
        if issparse(layer_data):
            layer_data = layer_data.toarray()
        elif isinstance(layer_data, np.matrix):
            layer_data = np.asarray(layer_data)
        self.X = torch.Tensor(layer_data)


        # Subsample if required
        if subsample_frac < 1:
            np.random.seed(42)
            n_to_keep = int(subsample_frac*len(self.X))
            indices = np.random.choice(range(len(self.X)), n_to_keep, replace=False)
            self.X = self.X[indices]
            adata = adata[indices]  
                    
        # Covariate to index
        self.id2cov = {}  # cov_name: dict_cov_2_id 
        self.Y_cov = {}   # cov: cov_ids
        
        for cov_name in covariate_keys:
            cov_ctrl = np.array(adata.obs[cov_name])
            unique_cov = np.unique(cov_ctrl)
            zip_cov_cat = dict(zip(unique_cov, np.arange(len(unique_cov))))
            self.id2cov[cov_name] = zip_cov_cat
            self.Y_cov[cov_name] = torch.tensor([zip_cov_cat[c] for c in cov_ctrl],dtype=torch.long)
        
        del adata
    




    def __getitem__(self, i):
        y = {cov: self.Y_cov[cov][i] for cov in self.Y_cov}
        X_i = self.X[i]
        X_norm = normalize_expression(X_i, X_i.sum(), self.encoder_type)
        return dict(X=X_i, X_norm=X_norm, y=y)





    def __len__(self):
        return len(self.X)  # assumes ctrl and pert have equal sample counts










