from torch.utils.data import DataLoader

from scduo.scduo_perturbation.ae.ed import *
import pandas as pd
import torch
import numpy as np
import random
import os
from scduo.scduo_perturbation import configs
import scanpy as sc
import episcanpy.api as epi
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
from torch.distributions import Normal, kl_divergence as kl
from scduo.scduo_perturbation.diffusion.diffusion import translate
matplotlib.use('Agg')


def get_input_dimensions(RNA_data):
    RNA_input_dim = RNA_data.var['highly_variable'].sum()
    return RNA_input_dim


def RNA_data_preprocessing(RNA_data, normalize_total=False, log1p=True, use_hvg=True, n_top_genes=3000):
    RNA_data.var_names_make_unique()

    if normalize_total:
        sc.pp.normalize_total(RNA_data)

    if log1p:
        sc.pp.log1p(RNA_data)

    if use_hvg:
        sc.pp.highly_variable_genes(RNA_data, n_top_genes=n_top_genes)
        RNA_data = RNA_data[:, RNA_data.var['highly_variable']]

    import os
    os.makedirs("/home/wuboyang/scduo-new/script/train_ae/outputs", exist_ok=True)
    RNA_data.write("/home/wuboyang/scduo-new/script/train_ae/outputs/RNA_data_hvg.h5ad")
    print(f"保存了高变基因筛选后的 AnnData ")


    return RNA_data


def preprocess_rna_data(file_path):
    # Step 1: Read RNA data
    RNA_data = sc.read_h5ad(file_path)

    # Step 2: Preprocess the data
    RNA_data = RNA_data_preprocessing(RNA_data)

    return RNA_data


def five_fold_split_dataset(
        RNA_data,
        seed=19193
):
    if not seed is None:
        setup_seed(seed)

    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)

    id_list = []
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))

    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count:]

        id_list.append([train_id, validation_id, test_id])

    return id_list


def save_dataset_ids(RNA_data, train_id, val_id, test_id, fold_idx):
    pd.DataFrame({
        "train_ids": RNA_data.obs_names[train_id]
    }).to_csv(f'/home/wuboyang/scduo-new/script/train_ae/outputs/train_ids.csv',
              index=False)

    pd.DataFrame({
        "val_ids": RNA_data.obs_names[val_id]
    }).to_csv(f'/home/wuboyang/scduo-new/script/train_ae/outputs/val_ids.csv',
              index=False)

    pd.DataFrame({
        "test_ids": RNA_data.obs_names[test_id]
    }).to_csv(f'/home/wuboyang/scduo-new/script/train_ae/outputs/test_ids.csv',
              index=False)


def _to_tensor(X):
    if sp.issparse(X):
        return torch.tensor(X.toarray(), dtype=torch.float32, device=configs.DEVICE)
    return torch.tensor(np.asarray(X), dtype=torch.float32, device=configs.DEVICE)
def convert_to_tensors(data_train, data_val, data_test):
    return _to_tensor(data_train.X), _to_tensor(data_val.X), _to_tensor(data_test.X)


def save_model(rna_encoder, rna_decoder, RNA_input_dim):
    torch.save({
        'rna_encoder_state_dict': rna_encoder.state_dict(),
        'rna_decoder_state_dict': rna_decoder.state_dict(),
        'RNA_input_dim': RNA_input_dim,
    }, f"/home/wuboyang/scduo-new/script/train_ae/outputs/model_info.pth")

    print("Model and dimensions saved.")


# def get_encoder_decoder(RNA_input_dim):
#     rna_encoder = NetBlock(
#         nlayer=2,
#         dim_list=[RNA_input_dim, 256, 128],
#         act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
#         dropout_rate=0.1,
#         noise_rate=0  #0.5
#     )

#     rna_decoder = NetBlock(
#         nlayer=2,
#         dim_list=[128, 256, RNA_input_dim],
#         act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
#         dropout_rate=0.1,
#         noise_rate=0
#     )

#     return rna_encoder, rna_decoder

from scduo.scduo_perturbation.ae.ed import MLP
def get_encoder_decoder(RNA_input_dim):
    rna_encoder = MLP(dims=[RNA_input_dim, 1000, 1000,128],
                      norm=True,
                      dropout=False,
                      dropout_p=0.0,
                      activation=nn.ReLU,
                      final_activation=None,
                      norm_type="batchnorm",
                      out_mult=2
                      )
    rna_decoder= MLP(dims=[128, 1000, 1000, RNA_input_dim],
                     norm=True,
                     dropout=False,
                     dropout_p=0.0,
                     activation=nn.ReLU,
                     final_activation="ReLU",
                     norm_type="batchnorm",
                     out_mult=1
                    )
    return rna_encoder, rna_decoder


def train_ed(rna_encoder, rna_decoder, 
             rna_tensor_train, 
             rna_tensor_val, 
             optimizer_rna, 
             R2R_train_epoch=100, 
             batch_size=configs.BATCH_SIZE, r_loss_fn=nn.MSELoss(),
             patience=20, fold_idx=1):
    # RNA-AE training
    rna_train_losses = []
    rna_val_losses = []
    print(f'Fold {fold_idx} - Starting RNA-AE training...')
    best_val_loss, patience_counter = float('inf'), 0

    for epoch in range(R2R_train_epoch):
        rna_encoder.train()
        rna_decoder.train()
        train_loss = 0
        dataloader = DataLoader(rna_tensor_train, batch_size=batch_size, shuffle=True, drop_last=False)
        for idx, rna_batch in enumerate(dataloader):
            rna_batch = rna_batch.to(configs.DEVICE)
            optimizer_rna.zero_grad()

            noise_rate = 0.1
            kl_weight = 5e-4
            x = rna_batch
            noise = torch.randn_like(x)
            x_noise = x + noise * noise_rate
            h = rna_encoder(x_noise)
            mu, logvar = torch.chunk(h, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std                   # ⭐ reparameterize
            x_hat = rna_decoder(z)

            loss = 0
            loss_kl = kl(
                Normal(mu, std),
                Normal(0, 1)
            ).sum(dim=1).mean()

            loss_rec = ((rna_batch - x_hat) ** 2).sum(dim=1).mean()

            loss = 0.5 * loss_rec + 0.5 * (loss_kl * kl_weight)
            
            # encoded = rna_encoder(rna_batch)
            # decoded = rna_decoder(encoded)
            # loss = r_loss_fn(decoded, rna_batch) 

            loss.backward()
            optimizer_rna.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / np.ceil(rna_tensor_train.size(0) / batch_size)
        rna_train_losses.append(avg_train_loss)

        rna_encoder.eval()
        rna_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            dataloader = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
            for idx, rna_batch in enumerate(dataloader):
                rna_batch = rna_batch.to(configs.DEVICE)
                kl_weight = 5e-4
                noise_rate = 0.1
                x = rna_batch
                noise = torch.randn_like(x)
                x_noise = x + noise * noise_rate
                h = rna_encoder(x_noise)
                mu, logvar = torch.chunk(h, 2, dim=1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std                   # ⭐ reparameterize
                x_hat = rna_decoder(z)

                loss = 0
                loss_kl = kl(
                Normal(mu, std),
                Normal(0, 1)
                ).sum(dim=1).mean()

                loss_rec = ((rna_batch - x_hat) ** 2).sum(dim=1).mean()

                loss = 0.5 * loss_rec + 0.5 * (loss_kl * kl_weight)
                # encoded = rna_encoder(rna_batch)
                # decoded = rna_decoder(encoded)
                # loss = r_loss_fn(decoded, rna_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / np.ceil(rna_tensor_val.size(0) / batch_size)
        rna_val_losses.append(avg_val_loss)

        print(f'Fold {fold_idx}, Epoch [{epoch + 1}/{R2R_train_epoch}], AE Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(rna_encoder.state_dict(), f'/home/wuboyang/scduo-new/script/train_ae/outputs/best_rna_encoder.pth')
            torch.save(rna_decoder.state_dict(), f'/home/wuboyang/scduo-new/script/train_ae/outputs/best_rna_decoder.pth')
            print("Validation loss decreased, saving model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f'Early stopping triggered.')
                break

    rna_encoder.load_state_dict(torch.load(f'/home/wuboyang/scduo-new/script/train_ae/outputs/best_rna_encoder.pth'))
    rna_decoder.load_state_dict(torch.load(f'/home/wuboyang/scduo-new/script/train_ae/outputs/best_rna_decoder.pth'))

    train_loss = np.mean(rna_train_losses) 
    val_loss = np.mean(rna_val_losses) 

    print(f'Pre-training for fold {fold_idx} complete.')

    return train_loss, val_loss, rna_train_losses, rna_val_losses

# def train_ed_enhanced(ep, rna_gen, atac_gen, rna_encoder, atac_encoder, rna_decoder, atac_decoder,
#              rna_tensor_train, atac_tensor_train,
#              rna_tensor_val, atac_tensor_val,
#              optimizer, batch_size=configs.BATCH_SIZE,
#              r_loss_fn=nn.MSELoss(), a_loss_fn=nn.BCELoss()):

#     print(f'Epoch {ep} - Starting AE fine-training...')

#     num_cells = rna_tensor_train.size(0)
#     indices = np.arange(num_cells)
#     np.random.shuffle(indices)
#     rna_tensor_train = rna_tensor_train[indices]
#     atac_tensor_train = atac_tensor_train[indices]

#     rna_gen.eval()
#     atac_gen.eval()
#     rna_encoder.train()
#     rna_decoder.train()
#     atac_encoder.train()
#     atac_decoder.train()
#     train_loss = 0
#     batches_sel = 1

#     dataloader1 = DataLoader(rna_tensor_train, batch_size=batch_size, shuffle=False, drop_last=False)
#     dataloader2 = DataLoader(atac_tensor_train, batch_size=batch_size, shuffle=False, drop_last=False)
#     for idx, (rna_batch, atac_batch) in enumerate(zip(dataloader1, dataloader2)):
#         if idx > batches_sel-1:
#             break
#         rna_batch = rna_batch.to(configs.DEVICE)
#         atac_batch = atac_batch.to(configs.DEVICE)
#         optimizer.zero_grad()

#         zr_0 = rna_encoder(rna_batch)
#         za_0 = atac_encoder(atac_batch)

#         with torch.no_grad():
#             zra_0 = translate(zr_0, atac_gen).to(configs.DEVICE) # translating RNA to ATAC
#             zra_0 = zra_0.view(zra_0.size(0), -1)
#             zar_0 = translate(za_0, rna_gen).to(configs.DEVICE) # translating ATAC to RNA
#             zar_0 = zar_0.view(zar_0.size(0), -1)

#         xr_0 = rna_decoder(zr_0)
#         xar_0 = rna_decoder(zar_0)
#         xa_0 = atac_decoder(za_0)
#         xra_0 = atac_decoder(zra_0)
#         loss_rna = r_loss_fn(xar_0, rna_batch)
#         print(loss_rna)
#         loss_atac =  a_loss_fn(xra_0, atac_batch)
#         print(loss_atac)
#         loss = loss_rna + loss_atac
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     avg_train_loss = train_loss / batches_sel

#     num_cells = rna_tensor_val.size(0)
#     indices = np.arange(num_cells)
#     np.random.shuffle(indices)
#     rna_tensor_val = rna_tensor_val[indices]
#     atac_tensor_val = atac_tensor_val[indices]


#     rna_encoder.eval()
#     rna_decoder.eval()
#     atac_encoder.eval()
#     atac_decoder.eval()
#     val_loss = 0
#     with torch.no_grad():
#         dataloader1 = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
#         dataloader2 = DataLoader(atac_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
#         for idx, (rna_batch, atac_batch) in enumerate(zip(dataloader1, dataloader2)):
#             if idx > batches_sel - 1:
#                 break
#             rna_batch = rna_batch.to(configs.DEVICE)
#             atac_batch = atac_batch.to(configs.DEVICE)
#             zr_0 = rna_encoder(rna_batch)
#             za_0 = atac_encoder(atac_batch)
#             xr_0 = rna_decoder(zr_0)
#             xa_0 = atac_decoder(za_0)

#             zra_0 = translate(zr_0, atac_gen).to(configs.DEVICE)
#             zra_0 = zra_0.view(zra_0.size(0), -1)
#             zar_0 = translate(za_0, rna_gen).to(configs.DEVICE)
#             zar_0 = zar_0.view(zar_0.size(0), -1)
#             xar_0 = rna_decoder(zar_0)
#             xra_0 = atac_decoder(zra_0)


#             loss_rna = r_loss_fn(xar_0, rna_batch)
#             print(loss_rna)
#             loss_atac = a_loss_fn(xra_0, atac_batch)
#             print(loss_atac)
#             loss = loss_rna + loss_atac
#             val_loss += loss.item()

#     avg_val_loss = val_loss / batches_sel
#     return avg_train_loss, avg_val_loss

# def evaluate_ae(rna_gen, atac_gen, rna_encoder, atac_encoder, rna_decoder, atac_decoder,
#              rna_tensor_val, atac_tensor_val, batch_size=configs.BATCH_SIZE,
#              r_loss_fn=nn.MSELoss(), a_loss_fn=nn.BCELoss()):

#     rna_gen.eval()
#     atac_gen.eval()
#     rna_encoder.eval()
#     rna_decoder.eval()
#     atac_encoder.eval()
#     atac_decoder.eval()
#     total_loss = 0

#     dataloader1 = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
#     dataloader2 = DataLoader(atac_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
#     for rna_batch, atac_batch in zip(dataloader1, dataloader2):
#         rna_batch = rna_batch.to(configs.DEVICE)
#         atac_batch = atac_batch.to(configs.DEVICE)

#         with torch.no_grad():
#             zr_0 = rna_encoder(rna_batch)
#             za_0 = atac_encoder(atac_batch)
#             xr_0 = rna_decoder(zr_0)
#             xa_0 = atac_decoder(za_0)

#             zra_0 = translate(zr_0, atac_gen).to(configs.DEVICE)
#             zra_0 = zra_0.view(zra_0.size(0), -1)
#             zar_0 = translate(za_0, rna_gen).to(configs.DEVICE)
#             zar_0 = zar_0.view(zar_0.size(0), -1)
#             xar_0 = rna_decoder(zar_0)
#             xra_0 = atac_decoder(zra_0)
#             loss_rna = r_loss_fn(xar_0, rna_batch)
#             loss_atac = a_loss_fn(xra_0, atac_batch)
#             loss = loss_rna + loss_atac
#             total_loss += loss.item()

#     avg_loss = total_loss / np.ceil(rna_tensor_val.size(0) / batch_size)

#     return avg_loss


def evaluate_model(rna_encoder, rna_decoder, rna_tensor_test, r_loss_fn, batch_size=configs.BATCH_SIZE):

    rna_encoder.eval()
    rna_decoder.eval()
    test_loss = 0
    with torch.no_grad():
        dataloader1 = DataLoader(rna_tensor_test, batch_size=batch_size, shuffle=False, drop_last=False)
        for rna_batch in dataloader1:
            rna_batch = rna_batch.to(configs.DEVICE)
            kl_weight = 5e-4
            noise_rate = 0.1
            x = rna_batch
            noise = torch.randn_like(x)
            x_noise = x + noise * noise_rate
            h = rna_encoder(x_noise)
            mu, logvar = torch.chunk(h, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std                   # ⭐ reparameterize
            x_hat = rna_decoder(z)

            loss = 0
            loss_kl = kl(
            Normal(mu, std),
            Normal(0, 1)
            ).sum(dim=1).mean()

            loss_rec = ((rna_batch - x_hat) ** 2).sum(dim=1).mean()

            loss = 0.5 * loss_rec + 0.5 * (loss_kl * kl_weight)
            # encoded = rna_encoder(rna_batch)
            # decoded = rna_decoder(encoded)
            # loss = r_loss_fn(decoded, rna_batch)
            test_loss += loss.item()
    avg_test_loss = test_loss / np.ceil(rna_tensor_test.size(0) / batch_size)
    print(f'RNA Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



