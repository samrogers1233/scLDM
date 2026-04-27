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
                 encoder_type,              
                 conditioning_covariate,
                 n_cat=None,
                 kl_weight=5e-4,             
                 noise_rate=0.1,          
                 ):
        super().__init__()
        # save hyperparameters
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
        dims_enc = [self.in_dim, *self.encoder_kwargs["dims"]]  # encoder emits 2*latent (mean and log-var)
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


    def encode(self, batch):
        h = self.encoder(batch["X_norm"].to(self.device))
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                 
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
        loss = 0.5 * loss_rec + 0.5 * (loss_kl * self.kl_weight)  
        self.log(f"{stage}/loss_rec", loss_rec,      prog_bar=True, on_epoch=True)
        self.log(f"{stage}/loss_kl",  loss_kl,  prog_bar=True, on_epoch=True)
        self.log(f"{stage}/loss",loss,     prog_bar=True, on_epoch=True)
        return loss, x_hat

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










