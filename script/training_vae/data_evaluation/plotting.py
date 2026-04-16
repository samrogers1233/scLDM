# import numpy as np
# import matplotlib.pyplot as plt
# from umap import UMAP

# # 1️⃣ 加载 latent 向量
# latent_path = "/home/wuboyang/scduo-main/script/data_evaluation/outputs/latent.npz"
# data = np.load(latent_path)

# z_ctrl = data["z_ctrl"]  # shape (N, latent_dim)
# z_pert = data["z_pert"]  # shape (N, latent_dim)，与 z_ctrl 一一对应

# # 2️⃣ 拼接数据 & 构造标签
# z_all = np.concatenate([z_ctrl, z_pert], axis=0)
# labels = np.array(["control"] * len(z_ctrl) + ["perturbed"] * len(z_pert))

# # 3️⃣ 使用 UMAP 降维
# umap = UMAP(n_components=2, random_state=0)
# z_2d = umap.fit_transform(z_all)

# # 4️⃣ 绘图
# plt.figure(figsize=(6, 5))
# colors = {"control": "steelblue", "perturbed": "indianred"}
# for group in np.unique(labels):
#     idx = labels == group
#     plt.scatter(z_2d[idx, 0], z_2d[idx, 1], s=5, c=colors[group], label=group, alpha=0.6)

# plt.title("Latent space (UMAP)")
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.legend()
# plt.tight_layout()
# plt.savefig("/home/wuboyang/scduo-main/script/data_evaluation/outputs/latent_umap.png", dpi=300)
# plt.show()




import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scanpy as sc
import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
#from scdiffusionX.utils import MMD, LISI, random_forest, norm_total
import pandas as pd
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli
import muon as mu
import yaml
import seaborn as sns  
from tqdm import tqdm
from scduo.sc.vae.data.data_loader import RNAseqLoader
from scduo.sc.vae.models.base.vae_model import EncoderModel
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# 加载 latent 向量
latent_data = np.load("/home/wuboyang/scduo-main/script/data_evaluation/outputs/latent.npz")
z_ctrl = torch.tensor(latent_data["z_ctrl"]).cuda()
z_pert = torch.tensor(latent_data["z_pert"]).cuda()


encoder_config = "/home/wuboyang/scduo-main/script/training_vae/configs/encoder/default.yaml"
dataset_path = '/home/wuboyang/scduo-main/dataset/processed_data/train_covid.h5ad'
covariate_keys = "celltype" #"leiden"
num_class = 22
ae_path = "/home/wuboyang/scduo-main/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v10.ckpt"



adata = sc.read(dataset_path)



# get size factor for encoder
dataset = RNAseqLoader(data_path=dataset_path,
                            layer_key='X_counts',
                            covariate_keys=[covariate_keys],
                            subsample_frac=1, 
                            encoder_type='learnt_autoencoder',
                            condition_key="condition",
                            control_value="control",
                            perturbed_value="severe COVID-19"
                        )
gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}



# 从 dataloader 中获取原始表达数据
# 假设你之前还保留着 dataset 实例：
X_ctrl_true = dataset.X["ctrl"].cuda()
X_pert_true = dataset.X["pert"].cuda()




# Load encoder and decoder
with open(encoder_config, 'r') as file:
    yaml_content = file.read()
autoencoder_args = yaml.safe_load(yaml_content)

# Initialize encoder
autoencoder_args['encoder_kwargs']['ctrl']['norm_type']='batchnorm'
autoencoder_args['encoder_kwargs']['pert']['norm_type']='batchnorm'
encoder_model = EncoderModel(in_dim=gene_dim,
                                    n_cat=adata.obs[covariate_keys].unique().shape[0],
                                    conditioning_covariate=covariate_keys, 
                                    encoder_type='proportions',
                                    **autoencoder_args)

# Load weights 
encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])


# 解码器（已加载模型 encoder_model）
encoder_model.eval()
encoder_model.cuda()
with torch.no_grad():
    sf_ctrl = X_ctrl_true.sum(1, keepdim=True)
    sf_pert = X_pert_true.sum(1, keepdim=True)

    X_ctrl_recon = encoder_model.decode_one(z_ctrl, mod="ctrl")
    X_pert_recon = encoder_model.decode_one(z_pert, mod="pert")

# 转换为 numpy 以便评估
Xc_true, Xc_pred = X_ctrl_true.cpu().numpy(), X_ctrl_recon.cpu().numpy()
Xp_true, Xp_pred = X_pert_true.cpu().numpy(), X_pert_recon.cpu().numpy()

# 计算逐细胞 Pearson & R²
def compute_metrics(X_true, X_pred):
    pearson_list, r2_list = [], []
    for i in range(X_true.shape[0]):
        true_row, pred_row = X_true[i], X_pred[i]
        if np.std(true_row) == 0:  # 避免 pearson nan
            continue
        pearson, _ = pearsonr(true_row, pred_row)
        r2 = r2_score(true_row, pred_row)
        pearson_list.append(pearson)
        r2_list.append(r2)
    return pearson_list, r2_list

pearson_ctrl, r2_ctrl = compute_metrics(Xc_true, Xc_pred)
pearson_pert, r2_pert = compute_metrics(Xp_true, Xp_pred)

# 打印结果
print("✅ Control group:")
print(f"  Pearson (mean ± std): {np.mean(pearson_ctrl):.4f} ± {np.std(pearson_ctrl):.4f}")
print(f"  R² (mean ± std):      {np.mean(r2_ctrl):.4f} ± {np.std(r2_ctrl):.4f}")

print("✅ Perturbed group:")
print(f"  Pearson (mean ± std): {np.mean(pearson_pert):.4f} ± {np.std(pearson_pert):.4f}")
print(f"  R² (mean ± std):      {np.mean(r2_pert):.4f} ± {np.std(r2_pert):.4f}")
