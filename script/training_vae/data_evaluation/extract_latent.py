# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import scanpy as sc
# import anndata as ad
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from scipy import stats
# #from scdiffusionX.utils import MMD, LISI, random_forest, norm_total
# import pandas as pd
# from torch.distributions import Normal
# from scvi.distributions import NegativeBinomial
# from torch.distributions import Poisson, Bernoulli
# import muon as mu
# import yaml
# import seaborn as sns  
# from tqdm import tqdm
# from scduo.sc.vae.data.data_loader import RNAseqLoader
# from scduo.sc.vae.models.base.vae_model import EncoderModel



# # set model and data path
# encoder_config = "/home/wuboyang/scduo-main/script/training_vae/configs/encoder/default.yaml"
# dataset_path = '/home/wuboyang/scduo-main/dataset/processed_data/train_covid.h5ad'
# covariate_keys = "celltype" #"leiden"
# num_class = 22
# ae_path = "/home/wuboyang/scduo-main/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v10.ckpt"



# adata = sc.read(dataset_path)



# # get size factor for encoder
# dataset = RNAseqLoader(data_path=dataset_path,
#                             layer_key='X_counts',
#                             covariate_keys=[covariate_keys],
#                             subsample_frac=1, 
#                             encoder_type='learnt_autoencoder',
#                             condition_key="condition",
#                             control_value="control",
#                             perturbed_value="severe COVID-19"
#                         )
# gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}

# # Load encoder and decoder
# with open(encoder_config, 'r') as file:
#     yaml_content = file.read()
# autoencoder_args = yaml.safe_load(yaml_content)

# # Initialize encoder
# autoencoder_args['encoder_kwargs']['ctrl']['norm_type']='batchnorm'
# autoencoder_args['encoder_kwargs']['pert']['norm_type']='batchnorm'
# encoder_model = EncoderModel(in_dim=gene_dim,
#                                     n_cat=adata.obs[covariate_keys].unique().shape[0],
#                                     conditioning_covariate=covariate_keys, 
#                                     encoder_type='learnt_autoencoder',
#                                     **autoencoder_args)

# # Load weights 
# encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
# encoder_model.eval()
# encoder_model.cuda()

# z_ctrl_list, z_pert_list = [], []

# with torch.no_grad():
#     for i in tqdm(range(len(dataset))):
#         batch = dataset[i]

#         # 控制组
#         if "ctrl" in batch["X_norm"]:
#             X_ctrl = batch["X_norm"]["ctrl"].unsqueeze(0).cuda()
#             _, _, z_c = encoder_model.encode_one(X_ctrl, mod="ctrl")
#             z_ctrl_list.append(z_c.cpu().numpy())

#         # 扰动组
#         if "pert" in batch["X_norm"]:
#             X_pert = batch["X_norm"]["pert"].unsqueeze(0).cuda()
#             _, _, z_p = encoder_model.encode_one(X_pert, mod="pert")
#             z_pert_list.append(z_p.cpu().numpy())

# # 拼接为矩阵
# z_ctrl = np.concatenate(z_ctrl_list, axis=0) if z_ctrl_list else None
# z_pert = np.concatenate(z_pert_list, axis=0) if z_pert_list else None

# # 保存为 .npz 文件
# np.savez("/home/wuboyang/scduo-main/script/data_evaluation/outputs/latent.npz",
#          z_ctrl=z_ctrl, z_pert=z_pert)


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

from scduo.scduo_perturbation.vae.data.data_loader import RNAseqLoader
from  scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel   # 确保路径正确

# =================== 路径配置 ===================
encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
dataset_path = '/home/wuboyang/scduo-new/dataset/cross_species_data/train_species.h5ad'
covariate_keys = "celltype"
ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v14.ckpt"

# =================== 加载数据 ===================
adata = sc.read(dataset_path)

# =================== 初始化 DataLoader ===================
dataset = RNAseqLoader(
    data_path=dataset_path,
    layer_key='X_counts',
    covariate_keys=[covariate_keys],
    subsample_frac=1,
    encoder_type='learnt_autoencoder',
    condition_key="condition",
    control_value="Control",
    perturbed_value="Hpoly.Day10"
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

# =================== 加载模型 ===================
with open(encoder_config, 'r') as file:
    autoencoder_args = yaml.safe_load(file)

# 更新模型参数
autoencoder_args['encoder_kwargs']['norm_type'] = 'batchnorm'


gene_dim = dataset.X.shape[1] 
encoder_model = EncoderModel(
    in_dim=gene_dim,
    n_cat=adata.obs[covariate_keys].unique().shape[0],
    conditioning_covariate=covariate_keys,
    encoder_type='learnt_autoencoder',
    **autoencoder_args
)

# 加载权重
encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
encoder_model.eval()

# =================== 推理并收集输出 ===================
all_x_hat = []
all_input = []
latent_list = []

with torch.no_grad():
    for batch in dataloader:

        # _,x_hat = encoder_model(batch)
        # z,mu,logvar = encoder_model.encode(batch)

        # x_hat,_,_ = encoder_model(batch)
        z,mu,logvar = encoder_model.encode(batch)
        x_hat=encoder_model.decode(z)

        
        latent_list.append(z.cpu())

        all_x_hat.append(x_hat.cpu())
        all_input.append(batch["X_norm"].cpu())

Y_pred = torch.cat(all_x_hat, dim=0).numpy()  
Y_true = torch.cat(all_input, dim=0).numpy()   
latent = torch.cat(latent_list, dim=0).numpy()

# =================== 评估函数 ===================
from sklearn.metrics import r2_score
def evaluate_reconstruction(all_input, all_mu_hat):
    # y_true = all_input[mod].flatten()
    # y_pred = all_mu_hat[mod].flatten()
    y_true = all_input.mean(axis=0)
    y_pred = all_mu_hat.mean(axis=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson R: {corr:.4f}")


# =================== 运行评估 ===================
evaluate_reconstruction(Y_true, Y_pred)



# # 保存为 npz 文件
# save_dir = "/home/wuboyang/scduo-main/script/training_vae/data_evaluation/outputs"
# os.makedirs(save_dir, exist_ok=True)
# np.savez_compressed(os.path.join(save_dir, "latent_ctrl.npz"), data=latent_dict["ctrl"])
# np.savez_compressed(os.path.join(save_dir, "latent_pert.npz"), data=latent_dict["pert"])

# print("已保存潜在空间表示")





#药物扰动数据集重构(pbmc)： MSE: 0.0003, MAE: 0.0113, R²: 0.9507, Pearson R: 0.9856
#基因扰动数据集重构(norman)： MSE: 0.0006, MAE: 0.0156, R²: 0.9484, Pearson R: 0.9804
#药物扰动数据集重构(covid)： MSE: 0.0003, MAE: 0.0094, R²: 0.9646, Pearson R: 0.9870
#药物扰动数据集重构(H.poly)：MSE: 0.0005, MAE: 0.0149, R²: 0.9753, Pearson R: 0.9918
#基因扰动数据集重构(adamson): MSE: 0.0007, MAE: 0.0156, R²: 0.9554, Pearson R: 0.9810
#基因扰动数据集重构(norman新)： MSE: 0.0005, MAE: 0.0119, R²: 0.9499, Pearson R: 0.9784