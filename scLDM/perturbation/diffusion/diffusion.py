import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from scLDM.perturbation import configs
from numpy import e
import os
import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scLDM.perturbation.ae.autoencoder import *
import sys
device = configs.DEVICE
batch_size = configs.BATCH_SIZE

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Returns a linear schedule of betas from start to end with an input timestep
    """
    return torch.linspace(start, end, timesteps)

def ddim_sampling_parameters(num_steps, eta=0.0):
    device = configs.DEVICE
    betas = linear_beta_schedule(timesteps=num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
    sigmas = eta * torch.sqrt(
        (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) *
        (1 - alphas_cumprod / alphas_cumprod_prev)
    ).to(device)
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sigmas': sigmas
    }




def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device=device, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)


# Define beta schedule
timesteps = configs.TIMESTEPS
betas = linear_beta_schedule(timesteps=timesteps)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
bce_loss_fn = nn.BCELoss()
distance_fn = nn.MSELoss()
LAMBDA = 15



def train_translators(ep, genA, optimizerA, data_loaderA, data_loaderB, classes, label_emb, iterations=100, batch_size=None, device=None, T=None):
    if batch_size is None:
        batch_size = configs.BATCH_SIZE
    if device is None:
        device = configs.DEVICE
    assert T is not None, "T must be provided for training"
    timesteps = T

    pbar = tqdm(range(iterations), desc=f'Training Epoch {ep}')
    diff_loss = 0

    genA.train()

    num_cells = data_loaderA.size(0)
    indices = np.arange(num_cells)
    np.random.shuffle(indices)
    data_loaderA = data_loaderA[indices]
    data_loaderB = data_loaderB[indices]
    classes = classes[indices]

    num_batches = data_loaderA.size(0) // batch_size
    for i in pbar:
        optimizerA.zero_grad()
        k = i % num_batches
        xA0 = data_loaderA[k*batch_size:(k+1)*batch_size].to(configs.DEVICE)
        xB0 = data_loaderB[k*batch_size:(k+1)*batch_size].to(configs.DEVICE)
        class_batch = classes[k*batch_size:(k+1)*batch_size].to(configs.DEVICE)

        xA0 = xA0.unsqueeze(1)
        xB0 = xB0.unsqueeze(1)

        emb = label_emb(class_batch).unsqueeze(1)
        tA = torch.randint(0, timesteps, (xA0.shape[0],), device=device).long()
        noiseA = torch.randn_like(xA0)
        xAtA = forward_diffusion_sample(xA0, tA, device, noiseA)


        predA = genA(torch.cat([xAtA, xB0, emb], dim=1), tA)


        diffusion_loss = distance_fn(predA, noiseA) 

        diffusion_loss.backward()
        optimizerA.step()

        diff_loss += diffusion_loss.item()
        pbar.set_description(
            f"Epoch {ep}-Step {i + 1}/{iterations}-Diff={round(diffusion_loss.item(), 4)}"
        )


    avg_diffusion_loss = diff_loss / iterations


    return avg_diffusion_loss


@torch.no_grad()
def translate(xA0, classes, label_emb, model, release_time=configs.RELEASE_TIME, num_samples=10, num_steps=500, eta=0.0):
    model.eval()
    device = xA0.device
    b = xA0.shape[0]
    emb = label_emb(classes).unsqueeze(1)
    if xA0.dim() == 2:
        xA0 = xA0.unsqueeze(1)

    sampling_params = ddim_sampling_parameters(num_steps=num_steps, eta=eta)

    xBt_samples = []

    for _ in tqdm(range(num_samples), desc="Translating Samples",file=sys.stdout, mininterval=0.5):
        xBt = torch.randn_like(xA0).to(device)

        for i in reversed(range(num_steps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            if i > release_time:
                xBt = translate_before_release(xA0, xBt, emb, t, model, sampling_params)
            else:
                noiseA = torch.randn_like(xA0)
                xAt = sampling_params['sqrt_alphas_cumprod'][i] * xA0 + \
                      sampling_params['sqrt_one_minus_alphas_cumprod'][i] * noiseA
                xAt, xBt = translate_after_release(xAt, xBt,emb, t, model, sampling_params)

        xBt_samples.append(xBt)

    xBt_avg = torch.mean(torch.stack(xBt_samples), dim=0)
    return xBt_avg

@torch.no_grad()
def translate_before_release(xA0, xBt, emb, t, model, sampling_params):
    device = xA0.device
    if xA0.dim() == 2:
        xA0 = xA0.unsqueeze(1)
    if xBt.dim() == 2:
        xBt = xBt.unsqueeze(1)

    time_step = t[0].item()

    alpha_t = sampling_params['alphas'][time_step]
    alpha_bar_t = sampling_params['alphas_cumprod'][time_step]
    alpha_bar_t_prev = sampling_params['alphas_cumprod_prev'][time_step]
    sqrt_alpha_bar_t = sampling_params['sqrt_alphas_cumprod'][time_step]
    sqrt_one_minus_alpha_bar_t = sampling_params['sqrt_one_minus_alphas_cumprod'][time_step]
    sigma_t = sampling_params['sigmas'][time_step]

    noiseA = torch.randn_like(xA0)
    xAt = sqrt_alpha_bar_t * xA0 + sqrt_one_minus_alpha_bar_t * noiseA

    eps = model(torch.cat([xBt, xA0, emb], dim=1), t)

    x0_pred = (xBt - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

    if time_step > 0:
        noise = torch.randn_like(xBt) if sigma_t > 0 else 0
        xBt_prev = sampling_params['sqrt_alphas_cumprod'][time_step - 1] * x0_pred + \
                   sampling_params['sqrt_one_minus_alphas_cumprod'][time_step - 1] * eps + \
                   sigma_t * noise
    else:
        xBt_prev = x0_pred

    return xBt_prev

@torch.no_grad()
def translate_after_release(xAt, xBt,emb, t, model, sampling_params):
    device = xAt.device
    time_step = t[0].item()

    sqrt_alpha_bar_t = sampling_params['sqrt_alphas_cumprod'][time_step]
    sqrt_one_minus_alpha_bar_t = sampling_params['sqrt_one_minus_alphas_cumprod'][time_step]
    sigma_t = sampling_params['sigmas'][time_step]

    eps_B = model(torch.cat([xBt, xAt,emb], dim=1), t)
    x0_pred_B = (xBt - sqrt_one_minus_alpha_bar_t * eps_B) / sqrt_alpha_bar_t
    if time_step > 0:
        noise_B = torch.randn_like(xBt) if sigma_t > 0 else 0
        xBt_prev = sampling_params['sqrt_alphas_cumprod'][time_step - 1] * x0_pred_B + \
                   sampling_params['sqrt_one_minus_alphas_cumprod'][time_step - 1] * eps_B + \
                   sigma_t * noise_B
    else:
        xBt_prev = x0_pred_B

    eps_A = model(torch.cat([xAt, xBt,emb], dim=1), t)
    x0_pred_A = (xAt - sqrt_one_minus_alpha_bar_t * eps_A) / sqrt_alpha_bar_t
    if time_step > 0:
        noise_A = torch.randn_like(xAt) if sigma_t > 0 else 0
        xAt_prev = sampling_params['sqrt_alphas_cumprod'][time_step - 1] * x0_pred_A + \
                   sampling_params['sqrt_one_minus_alphas_cumprod'][time_step - 1] * eps_A + \
                   sigma_t * noise_A
    else:
        xAt_prev = x0_pred_A

    return xAt_prev, xBt_prev



def evaluate_translators(genA, data_loaderA, data_loaderB,ep, batch_size=None, device=None, T=None):
    genA.eval()
    if batch_size is None:
        batch_size = configs.BATCH_SIZE
    if device is None:
        device = configs.DEVICE
    assert T is not None, "T must be provided for evaluation"
    timesteps = T
    total_diff_loss = 0
    count = 0

    with torch.no_grad():
        dataloader1 = DataLoader(data_loaderA, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloader2 = DataLoader(data_loaderB, batch_size=batch_size, shuffle=False, drop_last=True)
        for xA0, xB0 in zip(dataloader1,dataloader2):
            xA0 = xA0.to(configs.DEVICE)
            xB0 = xB0.to(configs.DEVICE)
            xA0 = xA0.unsqueeze(1)
            xB0 = xB0.unsqueeze(1)
            count += batch_size
            tA = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noiseA = torch.randn_like(xA0)
            xAtA = forward_diffusion_sample(xA0, tA, device, noiseA)
            predA = genA(torch.cat([xAtA, xB0], dim=1), tA)

            diffusion_loss = distance_fn(predA, noiseA) 
            total_diff_loss += diffusion_loss.item() * batch_size

        avg_diff_loss = total_diff_loss / count

        return avg_diff_loss



def test_translators(genA, genB, data_loaderA, data_loaderB):
    genA.eval()
    genB.eval()

    total_diff_loss = 0
    count = 0

    with torch.no_grad():
        dataloader1 = DataLoader(data_loaderA, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloader2 = DataLoader(data_loaderB, batch_size=batch_size, shuffle=False, drop_last=True)
        for xA0, xB0 in zip(dataloader1, dataloader2):
            xA0 = xA0.to(configs.DEVICE)
            xB0 = xB0.to(configs.DEVICE)
            xA0 = xA0.unsqueeze(1)
            xB0 = xB0.unsqueeze(1)
            count += batch_size

            tA = torch.randint(0, timesteps, (batch_size,), device=device).long()
            tB = torch.randint(0, timesteps, (batch_size,), device=device).long()

            noiseA = torch.randn_like(xA0)
            noiseB = torch.randn_like(xB0)

            xAtA = forward_diffusion_sample(xA0, tA, device, noiseA)
            xBtB = forward_diffusion_sample(xB0, tB, device, noiseB)
            xAtB = forward_diffusion_sample(xA0, tB, device, noiseA)
            xBtA = forward_diffusion_sample(xB0, tA, device, noiseB)


            predA = genA(torch.cat([xAtA, xB0], dim=1), tA)
            predB = genB(torch.cat([xBtB, xA0], dim=1), tB)

            diffusion_loss = configs.weight_rna_to_atac * distance_fn(predA, noiseA) + configs.weight_atac_to_rna * distance_fn(predB, noiseB)
            total_diff_loss += diffusion_loss.item() * batch_size

        avg_diff_loss = total_diff_loss / count

        return avg_diff_loss



