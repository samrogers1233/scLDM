# import torch
# import torch.nn.functional as F
# import copy
# import pytorch_lightning as pl

# from scvi.distributions import NegativeBinomial
# from torch.distributions import Poisson, Bernoulli
# from .utils import MLP


# class EncoderModel(pl.LightningModule):
#     def __init__(self,
#                  in_dim,
#                  encoder_kwargs,
#                  learning_rate,
#                  weight_decay,
#                  covariate_specific_theta,
#                  encoder_type,
#                  conditioning_covariate,
#                  n_cat=None,
#                  ):

#         super().__init__()
#         # Input dimension
#         self.in_dim = in_dim

#         # Initialize attributes 
#         self.encoder_kwargs = copy.deepcopy(encoder_kwargs)  # if multimodal, dictionary with arguments, one per effect
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.covariate_specific_theta = covariate_specific_theta
#         self.encoder_type = encoder_type
#         self.conditioning_covariate = conditioning_covariate
#         self.n_cat = n_cat
#         self.modality_list = list(self.encoder_kwargs.keys())

#         # Theta for the negative binomial parameterization of scRNA-seq
#         in_dim_rna = self.in_dim["ctrl"]
#         # Inverse dispersion
#         if not covariate_specific_theta:
#             self.theta = torch.nn.Parameter(torch.randn(in_dim_rna), requires_grad=True)
#         else:
#             self.theta = torch.nn.Parameter(torch.randn(n_cat, in_dim_rna), requires_grad=True)


#         # Modality specific part 
#         self.encoder = {}
#         self.decoder = {}
#         for mod in self.modality_list:
#             encoder_kwargs[mod]["dims"] = [self.in_dim[mod], *encoder_kwargs[mod]["dims"]]
#             self.encoder[mod] = MLP(**encoder_kwargs[mod])
#             encoder_kwargs[mod]["dims"] = encoder_kwargs[mod]["dims"][::-1]
#             self.decoder[mod] = MLP(**encoder_kwargs[mod])
#         self.encoder = torch.nn.ModuleDict(self.encoder)
#         self.decoder = torch.nn.ModuleDict(self.decoder)

#         # 保存配置
#         self.save_hyperparameters()

#     def forward(self, batch):
#         X = {mod: batch["X"][mod].to(self.device) for mod in batch["X"]}
#         size_factor = {}
#         for mod in X:
#             size_factor_mod = X[mod].sum(1).unsqueeze(1).to(self.device)
#             size_factor[mod] = size_factor_mod

#         # Conditioning covariate encodings
#         y = batch["y"][self.conditioning_covariate].to(self.device)

#         # Make the encoding multimodal
#         z = self.encode(batch)
#         mu_hat = self.decode(z, size_factor)

#         # Compute the negative log-likelihood of the data under the model
#         loss = 0
#         for mod in mu_hat:
#             # Negative Binomial log-likelihood
#             if not self.covariate_specific_theta:
#                 px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta))
#             else:
#                 px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta[y]))
#             loss -= px.log_prob(X[mod]).sum(1).mean()
#         return loss, mu_hat

#     def _step(self, batch, dataset_type):
#         X = {mod: batch["X"][mod].to(self.device) for mod in batch["X"]}
#         size_factor = {}
#         for mod in X:
#             size_factor_mod = X[mod].sum(1).unsqueeze(1).to(self.device)
#             size_factor[mod] = size_factor_mod

#         # Conditioning covariate encodings
#         y = batch["y"][self.conditioning_covariate].to(self.device)

#         # Make the encoding multimodal
#         z = self.encode(batch)
#         mu_hat = self.decode(z, size_factor)

#         # Compute the negative log-likelihood of the data under the model
#         loss = 0
#         for mod in mu_hat:
#             # Negative Binomial log-likelihood
#             if not self.covariate_specific_theta:
#                 px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta))
#             else:
#                 px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta[y]))
#             loss -= px.log_prob(X[mod]).sum(1).mean()

#         self.log(f'{dataset_type}/loss', loss, on_epoch=True, prog_bar=True)
#         return loss, mu_hat



#     def training_step(self, batch, batch_idx):
#         loss, _ = self._step(batch,"train")
#         #self.log("train/loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, _ = self._step(batch,"val")
#         #self.log("val/loss", loss, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(),
#                                  lr=self.learning_rate,
#                                  weight_decay=self.weight_decay)

#     def encode(self, batch):
#         z = {}
#         for mod in self.modality_list:
#             z_mod = self.encoder[mod](batch["X_norm"][mod].to(self.device))
#             z[mod] = z_mod    
#         return z

#     def decode(self, x, size_factor):
#         mu_hat = {}
#         for mod in self.modality_list:
#             x_mod = self.decoder[mod](x[mod])
#             #mu_hat_mod = F.softplus(x_mod) * size_factor[mod]
#             mu_hat_mod = F.softmax(x_mod, dim=1)  # for Poisson counts the parameterization is similar to RNA 
#             mu_hat_mod = mu_hat_mod * size_factor[mod]
#             mu_hat[mod] = mu_hat_mod
#         return mu_hat






#双VAE
# models/encoder_vae.py
import copy, torch, pytorch_lightning as pl
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl
from scvi.distributions import NegativeBinomial
from .utils import MLP      
import torch.nn as nn      

class EncoderModel(pl.LightningModule):
    def __init__(self,
                 in_dim,
                 encoder_kwargs,
                 learning_rate,
                 weight_decay,
                 covariate_specific_theta,
                 encoder_type,               # 仍保留，方便与旧 cfg 对齐
                 conditioning_covariate,
                 n_cat=None,
                 kl_weight=5e-4,             # ⭐ 新增：KL 权重 β
                 noise_rate=0.1,             # ⭐ 可选：输入加噪
                 ):
        super().__init__()
        # ——————————保存参数——————————
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.encoder_kwargs = copy.deepcopy(encoder_kwargs)
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.covariate_specific_theta = covariate_specific_theta
        self.conditioning_covariate   = conditioning_covariate
        self.encoder_type = encoder_type
        self.n_cat = n_cat
        self.modality_list = list(self.encoder_kwargs.keys())
        self.kl_weight = kl_weight
        self.noise_rate = noise_rate

        if not covariate_specific_theta:
            self.theta = torch.nn.Parameter(
                torch.randn(in_dim), requires_grad=True)
        else:
            self.theta = torch.nn.Parameter(
                torch.randn(n_cat, in_dim), requires_grad=True)


        self.encoder, self.decoder = {}, {}
        dims_enc = [self.in_dim, *self.encoder_kwargs["dims"]]  # ⭐ 输出 2*latent
        dims_dec = [ *self.encoder_kwargs["dims"][::-1], self.in_dim]

        self.encoder = MLP(dims=dims_enc,
                                    activation=nn.ReLU,
                                    final_activation=None,
                                    out_mult=2
                                    )
        self.decoder = MLP(dims=dims_dec,
                                    activation=nn.ReLU,
                                    final_activation="ReLU",                                   
                                    out_mult=1 
                                    )

        # self.encoder = torch.nn.ModuleDict(self.encoder)
        # self.decoder = torch.nn.ModuleDict(self.decoder)


    def encode(self, batch):
        h = self.encoder(batch["X_norm"].to(self.device))
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                   # ⭐ reparameterize
        return z, mu, logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, batch):
        x_noisy_dict = {}
        batch["X_norm"] = batch["X_norm"].to(self.device)
        x= batch["X_norm"]
        noise = torch.randn_like(x)
        x_noisy_dict = x + noise * self.noise_rate
        z, mu, logvar = self.encode({"X_norm": x_noisy_dict})
        x_hat = self.decode(z)
        loss_kl = 0
        loss_rec = 0
        std= torch.exp(0.5 * logvar)
        loss_kl = kl(
            Normal(mu, std),
            Normal(0, 1)
        ).sum(dim=1)
        loss_kl = loss_kl.mean()
        loss_rec=((batch["X_norm"] - x_hat) ** 2).sum(dim=1)
        loss_rec = loss_rec.mean()
        return x_hat, loss_rec, loss_kl

    def _step(self, batch, stage):
        loss_rec, loss_kl = 0, 0
        x_hat,loss_rec, loss_kl = self.forward(batch)
        loss = 0.5 * loss_rec + 0.5 * (loss_kl * self.kl_weight)  # ⭐ 总损失
        self.log(f"{stage}/loss_rec", loss_rec,      prog_bar=True, on_epoch=True)
        self.log(f"{stage}/loss_kl",  loss_kl,  prog_bar=True, on_epoch=True)
        self.log(f"{stage}/loss",loss,     prog_bar=True, on_epoch=True)
        return loss, x_hat

    # Lightning 钩子
    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "val")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay)










