import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import scanpy as sc
import torch
import torch.optim as optim
import muon as mu
import yaml
import os


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
from scipy.sparse import issparse



encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
dataset_path = '/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad'
covariate_keys = "cell_type" 
num_class = 7
ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v2.ckpt"
adata = sc.read_h5ad(dataset_path)


# load autoencoder
with open(encoder_config, 'r') as file:
    yaml_content = file.read()
autoencoder_args = yaml.safe_load(yaml_content)

# Initialize encoder
encoder_model = EncoderModel(
    in_dim=adata.shape[1],
    n_cat=adata.obs[covariate_keys].unique().shape[0],
    conditioning_covariate=covariate_keys,
    encoder_type='learnt_autoencoder',
    **autoencoder_args
)

# Load weights 
encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])



defaults = dict(
    clip_denoised=True,
    batch_size=64,
    sample_fn="ddim",
    multimodal_model_path="/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs6/train_outputs/model200000.pt",
    output_dir="test",
    classifier_scale=0,
    devices='0',
    is_strict=True,
    all_save_num= 1024,
    seed=42,
    load_noise="",
    data_dir=dataset_path,
    condition='cell_type',
)
defaults.update(model_and_diffusion_defaults())
parser = argparse.ArgumentParser()
defaults['ctrl_dim'] = '1,100'
defaults['pert_dim'] = '1,100'
defaults['num_channels'] = 128
defaults['num_res_blocks'] = 1
defaults['resblock_updown'] = True
defaults['num_class'] = 7
defaults['class_cond'] = True
add_dict_to_argparser(parser, defaults)


# load diffusion backbone
args = parser.parse_known_args()[0]
args.ctrl_dim = [int(i) for i in args.ctrl_dim.split(',')]
args.pert_dim = [int(i) for i in args.pert_dim.split(',')]

adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
adata_pert = adata[adata.obs["condition"] == "stimulated"].copy()

dist_util.setup_dist(args.devices)

print("creating model and diffusion...")
multimodal_model, multimodal_diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
)
multimodal_model.load_state_dict_(
        dist_util.load_state_dict(args.multimodal_model_path, map_location="cpu"), is_strict=args.is_strict
)
multimodal_model.to(dist_util.dev())
optimizer2 = optim.Adam(multimodal_model.parameters(), lr=0.001)

# mdata = mu.read_h5mu(args.data_dir)
from sklearn.preprocessing import LabelEncoder
labels = adata_pert.obs[args.condition].values
label_encoder = LabelEncoder()
label_encoder.fit(labels)
classes_all = label_encoder.transform(labels)


type_list = np.array(['B', 'CD4T', 'CD8T', 'CD14+Mono', 'Dendritic', 'FCGR3A+Mono', 'NK'])


# calculate attention maps in each layers
time_step = 1
down_sample = 2
model_kwargs = {}
batch = {}


ctrl = []
pert = []
bs = 1000

batch_num = int(adata_ctrl.shape[0]/bs)+1
for i in range(batch_num):
    batch = {}
    ctrl_X = adata_ctrl[i*bs:(i+1)*bs].X
    pert_X = adata_pert[i*bs:(i+1)*bs].X
    if issparse(ctrl_X):
        ctrl_X = ctrl_X.toarray()
    if issparse(pert_X):
        pert_X = pert_X.toarray()
    ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
    pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
    from scduo.scduo_perturbation.vae.data.utils import normalize_expression
    ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
    pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
    all_norm = torch.cat([ctrl_X_norm, pert_X_norm], dim=0)
    batch["X_norm"] = all_norm
    half=ctrl_X.shape[0]
    z,_,_ = encoder_model.encode(batch)
    ctrl.append(z[:half])
    pert.append(z[half:])
# rescaling into std = 1
ctrl = torch.concat(ctrl).to(dtype=torch.float32, device=dist_util.dev())
pert = torch.concat(pert).to(dtype=torch.float32, device=dist_util.dev())
npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor3.npz')
ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
pert_std10 = float(np.asarray(npzfile["pert_std"]))

with torch.no_grad():
    audio_cond_all = ((ctrl / ctrl_std10).unsqueeze(1)).cpu().numpy().astype(np.float32)
            
audio_start = pert.unsqueeze(1).to(dist_util.dev())
model_kwargs["label"] = torch.tensor(classes_all).to(dist_util.dev())
model_kwargs["audio_cond"] = torch.tensor(audio_cond_all).to(dist_util.dev())
noise ={"audio":torch.randn_like(audio_start)}

#0 means t_th step, 1 means the audio gives groundtruth, 2 means the video gives the groundtruth
# condition_index = x_start["condition"]  
t = (torch.ones(audio_start.shape[0], device=dist_util.dev())*time_step).to(dtype=torch.int)

audio_t = multimodal_diffusion.q_sample(audio_start, t, noise = noise["audio"])#.detach()

att_layer1 = []
att_layer2 = []
att_layer3 = []

for celltype in type_list:
    index = (adata_pert.obs['cell_type'] == celltype)
    # sample_id = np.random.choice(np.arange(0, index.sum()), size=22, replace=False)
    audio_t_i = audio_t[index]
    t_i = t[index]
    labels = model_kwargs["label"][index]
    audio_conds = model_kwargs["audio_cond"][index]
    noise_pred_video, att_maps = multimodal_model(audio_t_i,t_i,labels,audio_conds,return_attvec=True)
    att_layer1.append(att_maps[1])
    att_layer2.append(att_maps[3])
    att_layer3.append(att_maps[5])


# fine the key elements in gene and peak. 14 is the target cell type index (CD4+ T activated here)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(12,12))
plt.imshow(att_layer2[6].mean(0).cpu().detach().numpy(),cmap='coolwarm')
plt.ylabel('gene')
plt.xlabel('gene')
out = "/home/wuboyang/scduo-new/figures/att_layer2_celltype1.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
print(f"Saved figure to: {out}")




plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
cross_map = att_layer2[6].mean(0).cpu().detach().numpy()
flattened_indices = np.argsort(cross_map, axis=None)[-10:]
positions = np.unravel_index(flattened_indices, cross_map.shape)
max_values = cross_map[positions]
print("The position (x,y) of the top10 elements:")
print('x(peak):', positions[0])
print('y(gene):', positions[1])
print("Top 10 values:", max_values)



plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
key_col = np.zeros((len(att_layer2),att_layer2[0].shape[-1]))
for i in range(len(att_layer2)):
    for j in range(att_layer2[i].shape[0]):
        att_mean = att_layer2[i][j].mean(0).detach().cpu()
        key_c = np.where(att_mean>0.35)[0]
        for c in key_c:
            key_col[i,c] += 1
    key_col[i] = key_col[i]/att_layer2[i].shape[0]

plt.figure(figsize=(32,8))
sns.heatmap(key_col, cmap='coolwarm', vmax=0.2)
plt.yticks(ticks=np.arange(type_list.shape[0]), labels=type_list)
plt.xticks(ticks=np.arange(0,key_col.shape[1],5), labels=np.arange(0,key_col.shape[1],5))
plt.title('what RNA element to focus')
out = "/home/wuboyang/scduo-new/figures/gene_element.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
print(f"Saved figure to: {out}")