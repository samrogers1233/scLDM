import numpy as np
import torch, random, os
from torch.optim import Adam
from tqdm import tqdm
import yaml
from scipy.sparse import issparse
import scduo.scduo_perturbation.configs as configs
from scduo.scduo_perturbation.diffusion.AttnUnet import Unet
from scduo.scduo_perturbation.diffusion.PlainUnet import SimpleUnet_plain
from scduo.scduo_perturbation.ae.autoencoder import get_encoder_decoder
from scduo.scduo_perturbation.diffusion.diffusion import *
from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = configs.DEVICE

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(plain=False):
    if not plain:
        model = Unet(
            configs.DIM,
            channels=3,     # 仍然保留2通道：你的 train_translators 会 cat([x_t, xB0], dim=1)
            out_dim=1,
            dim_mults=(1, 2, 4, 8, 8),
        )
    else:
        model = SimpleUnet_plain(in_dim=1, dim=64, out_dim=1)
    print("Num params:", sum(p.numel() for p in model.parameters()))
    return model

# ---------- 新增：单 VAE 场景的 encode 工具 ----------
@torch.no_grad()
def encode_raw_data(encoder_model, adata_ctrl):
    ctrl = []
    bs = 100

    batch_num = int(adata_ctrl.shape[0]/bs)+1
    for i in range(batch_num):
        batch = {}
        ctrl_X = adata_ctrl[i*bs:(i+1)*bs].X

        if issparse(ctrl_X):
            ctrl_X = ctrl_X.toarray()
        
        ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
        from scduo.scduo_perturbation.vae.data.utils import normalize_expression
        ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
        batch["X_norm"] = torch.tensor(ctrl_X_norm, dtype=encoder_model.dtype)

        z,_,_ = encoder_model.encode(batch)
        ctrl.append(z)

    ctrl = torch.concat(ctrl)
    ctrl_std10 = ctrl.std(0).mean()*10
    return (ctrl/ctrl_std10)
        

# ---------- 核心：单模型训练循环 ----------
def train_with_early_stopping_single(
    gen, optimizer_gen,
    encoder_model,                      # 已训练好的 VAE 编码器/解码器（只用来提特征）
    ctrl_train, pert_train,                # ctrl/pert 训练数据（可为 DataLoader 或 tensor）
    ctrl_val, pert_val,                    # 验证                  
    num_epochs=150, patience=50, iterations_per_epoch=100,
    classes=None, label_emb=None
):
    # 先离线编码到潜在空间（与原逻辑一致：translator 在潜空间中学习）
    with torch.no_grad():
        z_ctrl_train = encode_raw_data(encoder_model, ctrl_train)
        z_pert_train = encode_raw_data(encoder_model, pert_train)
        z_ctrl_val   = encode_raw_data(encoder_model, ctrl_val)
        z_pert_val   = encode_raw_data(encoder_model, pert_val)


    best_val = float('inf'); wait = 0
    diff_train_losses, diff_val_losses = [], []

    save_dir = '/home/wuboyang/scduo-new/script/training_diffusion/outputs'
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_gen2.pth')

    print('training translator (single-model) ......')
    for ep in range(1, num_epochs + 1):
        # 你的 train_translators 逻辑保持不变（单模型版本）
        avg_train = train_translators(ep, gen, optimizer_gen, z_pert_train, z_ctrl_train, classes, label_emb,
                                      iterations=iterations_per_epoch, batch_size=configs.BATCH_SIZE, device=device, T=configs.TIMESTEPS)
        diff_train_losses.append(avg_train)

        # 评估函数请改成单模型签名：evaluate_translators(gen, z_ctrl_val, z_pert_val, ep)
        # avg_val   = evaluate_translators(gen, z_pert_val, z_ctrl_val, ep, 
        #                                   batch_size=configs.BATCH_SIZE, device=device, T=configs.TIMESTEPS)
        # diff_val_losses.append(avg_val)


        print(f"Epoch {ep}: Train {avg_train:.4f}  ")

        # early stopping
        if avg_train < best_val:
            best_val = avg_train; wait = 0
            torch.save(gen.state_dict(), best_path)
            print("Validation improved, saved:", best_path)
        else:
            wait += 1
            print(f"No improv. for {wait} epochs")
            if wait >= patience:
                print("Early stopping.")
                break

    # 加载最佳
    gen.load_state_dict(torch.load(best_path, map_location=device))
    return gen, (diff_train_losses, diff_val_losses)

# ----------------- main -----------------
if __name__ == "__main__":
    setup_seed(19193)

    train_dataset_path = "/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad"
    val_dataset_path = "/home/wuboyang/scduo-new/dataset/processed_data/valid_pbmc.h5ad"
    condition = 'cell_type'
    condition_key = "condition"
    control_value="control"
    perturbed_value="stimulated" 
    embedding_dim = 128
    train_data= sc.read(train_dataset_path)
    val_data= sc.read(val_dataset_path)
    ctrl_train_loader = train_data[train_data.obs[condition_key] == control_value].copy()
    pert_train_loader = train_data[train_data.obs[condition_key] == perturbed_value].copy()
    ctrl_val_loader   = val_data[val_data.obs[condition_key] == control_value].copy()
    pert_val_loader   = val_data[val_data.obs[condition_key] == perturbed_value].copy()
    
    labels = ctrl_train_loader.obs[condition].values
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    classes= label_encoder.transform(labels)
    num_classes = ctrl_train_loader.obs[condition].nunique()
    if num_classes is not None:
        label_emb = nn.Embedding(num_classes, embedding_dim).to(configs.DEVICE)
    classes = torch.tensor(classes, dtype=torch.long, device=configs.DEVICE)

    E = np.load("/home/wuboyang/scduo-new/script/training_diffusion/outputs/embedding/celltype_emb.npz")["emb"]
    E = torch.as_tensor(E, dtype=torch.float32, device=configs.DEVICE)
    with torch.no_grad():
        label_emb.weight.copy_(E)            
        label_emb.weight.requires_grad_(False)    

    encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
    dataset_path = '/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad'
    covariate_keys = "cell_type" 
    num_class = 6
    ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v1.ckpt"
    adata = sc.read(dataset_path)

    with open(encoder_config, 'r') as file:
            yaml_content = file.read()
    autoencoder_args = yaml.safe_load(yaml_content)
    encoder_model = EncoderModel(in_dim=adata.shape[1],
                                        n_cat=num_class,
                                        conditioning_covariate="cell_type", 
                                        encoder_type='learnt_autoencoder',
                                        **autoencoder_args)
    
    encoder_model.load_state_dict(torch.load(ae_path)["state_dict"])
    encoder_model.eval()

    gen = get_model().to(device)
    optim_gen = Adam(gen.parameters(), lr=configs.GENA_LR)

    # 4) 开训
    best_gen, (diff_tr, diff_val) = train_with_early_stopping_single(
        gen, optim_gen,
        encoder_model,
        ctrl_train_loader, pert_train_loader,
        ctrl_val_loader,   pert_val_loader,
        num_epochs=configs.EPOCHS,
        patience=50,
        iterations_per_epoch=100,
        classes=classes, label_emb =label_emb,
    )
