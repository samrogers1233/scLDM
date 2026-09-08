"""
This code is extended from guided_diffusion: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
Multi-modal guassian have been added, as well as zero-shot conditional generation.
"""

import enum
import math
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from . import dist_util


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
         
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}
            
        B = x["audio"].shape[0]
        assert t.shape == (B,)
        
        audio_output = model(x["audio"], self._scale_timesteps(t), **model_kwargs) # when ddim, t is not mapped
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        def get_variance(model_output, x):
            if x.dim() == 3:
                dim=1
                
            elif x.dim() == 5 :
                dim=2
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                assert model_output.shape[dim] == x.shape[dim] * 2
                model_output, model_var_values = th.split(model_output, x.shape[dim], dim=dim)
                if self.model_var_type == ModelVarType.LEARNED:
                    model_log_variance = model_var_values
                    model_variance = th.exp(model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x.shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                    # The model_var_values is [-1, 1] for [min_var, max_var].
                    frac = (model_var_values + 1) / 2
                    model_log_variance = frac * max_log + (1 - frac) * min_log
                    model_variance = th.exp(model_log_variance)
            else:
                model_variance, model_log_variance = {
                    # for fixedlarge, we set the initial (log-)variance like so
                    # to get a better decoder log likelihood.
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                model_variance = _extract_into_tensor(model_variance, t, x.shape)
                model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
                )
                model_mean = model_output
            elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = process_xstart(model_output)

                else:
                    '''
                    if the model predicts the epsilon, pred xstart from predicted eps, then pred x_{t-1}
                    '''
                    pred_xstart = process_xstart(
                        self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                    )
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=x, t=t
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
            )
            return model_mean, model_variance, model_log_variance, pred_xstart
    
        audio_mean, audio_variance, audio_log_variance, pred_audio_xstart = get_variance(audio_output, x["audio"])

        return {
            "mean": { "audio": audio_mean},
            "variance": { "audio": audio_variance},
            "log_variance": { "audio": audio_log_variance},
            "pred_xstart": { "audio":pred_audio_xstart},
            "model_predict":{ "audio": audio_output}
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean
    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise=None
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        audio_noise = th.randn_like(x["audio"])
        
        audio_nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x["audio"].shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
   
        audio_sample = out["mean"]["audio"] + audio_nonzero_mask * th.exp(0.5 * out["log_variance"]["audio"]) * audio_noise

        return {"sample": {"audio":  audio_sample}, \
            "pred_start": {"audio": out["pred_xstart"]["audio"]},
            "pred_noise":{"audio": out["model_predict"]["audio"]}}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        gen_mode='atac',
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress
        ):
            final = sample
        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False
    ):
        if device is None:
            device = dist_util.dev()
                
    
        audio = th.randn(*shape["audio"], device='cpu')
        audio = audio.to(device)
        x = {"audio":audio}
        indices = list(range(self.num_timesteps))[::-1] # 0 to 999
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        for i in indices:
            cond = None
            timestep = self.timestep_map[i]
            if cond_fn is not None: 
                cond= cond_fn
            

            t = th.tensor([i] * shape["audio"][0], device=device)
                
            with th.no_grad():
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                yield out["sample"]
                x = out["sample"]
    
    def conditional_p_sample_loop(
        self,
        model,
        shape,
        use_fp16,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        class_scale=0.
    ):
        final = None
        if class_scale == 0:
            conditional_p_sample_loop_progressive_func = self.conditional_p_sample_loop_progressive_unscale
        else:
            conditional_p_sample_loop_progressive_func = self.conditional_p_sample_loop_progressive_scale

        for sample in conditional_p_sample_loop_progressive_func(
            model,
            shape,
            use_fp16,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            class_scale=class_scale
        ):
            final = sample

        return final


    def conditional_p_sample_loop_progressive_unscale(
        self,
        model,
        shape,
        use_fp16,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        class_scale=0.0,
    ):
        if device is None:
            device = dist_util.dev()
        
        if noise is None:
            audio = th.randn(*shape["audio"], device='cpu')
            audio = audio.to(device)
            noise = {"audio":audio}
       
        x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        audio_condition= None

        if "audio" in model_kwargs.keys():
            audio_condition = model_kwargs.pop("audio")

        for i in indices:
            cond = None
            timestep = self.timestep_map[i]
            if cond_fn is not None: 
                cond = cond_fn

            t = th.tensor([i] * shape["audio"][0], device=device)
           
            if audio_condition is not None:
                x["audio"] = self.q_sample(audio_condition, t, noise = noise["audio"])

                            
            with th.no_grad():
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                yield out["sample"]
                x = out["sample"]   
                
    def conditional_p_sample_loop_progressive_scale(
        self,
        model,
        shape,
        use_fp16,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        class_scale=3.0,
        
    ):
      
        if device is None:
            device = dist_util.dev()
        
        if noise is None:
            audio = th.randn(*shape["audio"], device='cpu')
            audio = audio.to(device)
            noise = {"audio":audio}
       
        x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        audio_condition= None

        if "audio" in model_kwargs.keys():
            audio_condition = model_kwargs.pop("audio")
        
        for i in indices:
            cond = None
            timestep = self.timestep_map[i]
            if cond_fn is not None: 
                cond = cond_fn

            t = th.tensor([i] * shape["audio"][0], device = device)
            # first get unconditional generation results
            
            if audio_condition is not None:
                condition = "audio"
                target = "video"
                x[condition] = self.q_sample(audio_condition, t, noise = noise[condition])
                previous_step_condition = self.q_sample(audio_condition, t-1, noise = noise[condition])
            
            with th.enable_grad():                
                none_zero_mask = (t != 0).float().view(-1, *([1] * (len(x[target].shape) - 1)))
                x[target] = x[target].detach().requires_grad_()
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                
                previous_step_pred = out["sample"]
                
                loss = mean_flat((previous_step_pred[condition] - previous_step_condition) ** 2)
                loss_scale = 1.
                if use_fp16 == True:
                    loss_scale = 2 ** 20
                grad = th.autograd.grad(loss.mean()* loss_scale , x[target])[0]
                # print(f"!!!!!!!!!!!!!!!grad:{grad.sum()}")
                x[target] = previous_step_pred[target] - none_zero_mask * grad* class_scale * self.sqrt_alphas_cumprod[i]              
                            
            yield x
 
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = {}
        eps["audio"] = self._predict_eps_from_xstart(x["audio"], t, out["pred_xstart"]["audio"])

        alpha_bar = {}
        alpha_bar["audio"] = _extract_into_tensor(self.alphas_cumprod, t, x["audio"].shape)

        alpha_bar_prev = {}
        alpha_bar_prev["audio"] = _extract_into_tensor(self.alphas_cumprod_prev, t, x["audio"].shape)
        
        sigma = {}
        sigma["audio"] = (
            eta * th.sqrt((1 - alpha_bar_prev["audio"]) / (1 - alpha_bar["audio"]))
            * th.sqrt(1 - alpha_bar["audio"] / alpha_bar_prev["audio"])
        )

        # Equation 12.
        noise = {}
        noise["audio"] = th.randn_like(x["audio"])

        mean_pred = {}
        mean_pred["audio"] = (
            out["pred_xstart"]["audio"] * th.sqrt(alpha_bar_prev["audio"])
            + th.sqrt(1 - alpha_bar_prev["audio"] - sigma["audio"] ** 2) * eps["audio"]
        )

        nonzero_mask = {}
        nonzero_mask["audio"] = (
            (t != 0).float().view(-1, *([1] * (len(x["audio"].shape) - 1)))
        )  # no noise when t == 0

        sample = {}
        sample["audio"] = mean_pred["audio"] + nonzero_mask["audio"] * sigma["audio"] * noise["audio"]

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = {}
        eps["audio"] = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["audio"].shape) * x["audio"]
            - out["audio"]["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["audio"].shape)

        alpha_bar_next = {}
        alpha_bar_next["audio"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["audio"].shape)

        # Equation 12. reversed
        mean_pred = {}
        mean_pred["audio"] = (
            out["pred_xstart"]["audio"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps["audio"]
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        eta=0.0,
        gen_mode='atac',
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            gen_mode=gen_mode,
        ):
            final = sample
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        gen_mode='atac',
    ):
        if device is None:
            device = dist_util.dev()
       
        # caution: video=RNA, audio=ATAC

        audio = th.randn(*shape["audio"], device='cpu')
        audio = audio.to(device)
        x = {"audio":audio}

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape["audio"][0], device=device)
            cond=None
            timestep = self.timestep_map[i]
            if cond_fn is not None:
                cond = cond_fn

            with th.no_grad():
                if noise is not None:
                    noise_t = self.q_sample(noise, t)
                    # noise_t = noise
                    if gen_mode=='pert':
                        x['video'] = noise_t.unsqueeze(1)
                    elif gen_mode=='ctrl':
                        x['audio'] = noise_t.unsqueeze(1)
                    else:
                        NotImplementedError(f"not support gen mode: {gen_mode}")
                out = self.ddim_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out["sample"]
                x = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        audio_true_mean, _, audio_true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start["audio"], x_t=x_t["audio"], t=t
        )
        true_mean = {"audio": audio_true_mean}
        true_log_variance_clipped = {"audio": audio_true_log_variance_clipped}
        
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
     
        kl = {}
        decoder_nll={}
        output = {}
        for key in ["audio"]:
            kl[key] = normal_kl(
            true_mean[key], true_log_variance_clipped[key], out["mean"][key], out["log_variance"][key]
        )
            kl[key] = mean_flat(kl[key]) / np.log(2.0)

            decoder_nll[key] = -discretized_gaussian_log_likelihood(
            x_start[key], means=out["mean"][key], log_scales=0.5 * out["log_variance"][key]
        )
            assert decoder_nll[key].shape == x_start[key].shape
            decoder_nll[key] = mean_flat(decoder_nll[key]) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            output[key] = th.where((t == 0), decoder_nll[key], kl[key])
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def predict_image_qt_t_step(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        return x_t

    def multimodal_training_losses(self, model, x_start, t,  model_kwargs=None, noise=None):
        audio_start = x_start['audio']
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise ={"audio":th.randn_like(audio_start)}
        
        #0 means t_th step, 1 means the audio gives groundtruth, 2 means the video gives the groundtruth
        # condition_index = x_start["condition"]  
        audio_t = self.q_sample(audio_start, t, noise = noise["audio"])
  
        audio_output = model(audio_t, ts=t,  **model_kwargs)
     
        audio_loss = {}
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
                
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                  
                
                audio_output, audio_var_values = th.split(audio_output, audio_start.shape[1], dim=1)
                    # Learn the variance using the variational bound, but don't let
                    # it affect our mean prediction.
                audio_frozen_out = th.cat([audio_output.detach(), audio_var_values], dim=1)
                frozen_out = {"audio": audio_frozen_out}
                x_t = {"audio": audio_t}
                vb_loss = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: [r["audio"]],
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                audio_loss["vb"] = vb_loss["audio"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    audio_loss["vb"] *= self.num_timesteps / 1000.0
            audio_target = {
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=audio_start, x_t=audio_t, t=t
                    )[0],
                    ModelMeanType.START_X: audio_start,
                    ModelMeanType.EPSILON: noise["audio"],   # noise
                }[self.model_mean_type]     
               
            audio_loss["mse"] = mean_flat((audio_target - audio_output) ** 2)
            
        term = {"loss":0}
 
        for key in audio_loss.keys():
            term[f"{key}_audio"] =  audio_loss[key]
            # term[f"{key}_all"] = video_mask * video_loss[key] + audio_mask * audio_loss[key]
            term["loss"] += term[f"{key}_audio"]  ##
  
        return term


    def _motion_variance(self, predict, target):
       
        assert predict.shape==target.shape
        predict_motion = predict[:,1:,...]-predict[:,:-1,...]
        target_motion = target[:,1:,...]-target[:,:-1,...]
        return  0.05*mean_flat((predict_motion - target_motion) ** 2) 
        
    def _prior_bpd(self, x_start):
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
