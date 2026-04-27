"""
Sample gene-perturbation latents from a trained single-diffusion model.

Two modes via --mode:
  iid  : perturbed cells come from the same dataset as control (cp_type=="stimulated").
  ood  : perturbed cells come from a separate OOD file given by --ood_data_dir.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math

import numpy as np
import torch
import torch as th
import torch.distributed as dist

import scanpy as sc
import yaml
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
import ot

from scLDM.perturbation.diffusion import dist_util, logger
from scLDM.perturbation.diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from scLDM.perturbation.diffusion.common import set_seed_logger_random
from scLDM.perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from scLDM.perturbation.vae.models.base.vae_model import EncoderModel
from scLDM.perturbation.vae.data.utils import normalize_expression


def load_gene2vec_txt_simple(path, dtype=np.float32):
    with open(path, "r") as f:
        num_genes, dim = map(int, f.readline().strip().split())
        genes = []
        embs = np.zeros((num_genes, dim), dtype=dtype)
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            genes.append(parts[0])
            embs[i] = np.asarray(parts[1:], dtype=dtype)
    gene2idx = {g: i for i, g in enumerate(genes)}
    return genes, embs, gene2idx


def build_perturbation_embeddings(adata_pert, gene2vec_path, perturbation_key="condition"):
    """Per-cell gene2vec embeddings. Combo labels joined by '_' are averaged."""
    _, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
    dim = embs.shape[1]
    pert_series = adata_pert.obs[perturbation_key].astype(str)

    vectors, missing = [], []
    for g in pert_series:
        tokens = g.split("_")
        vec = np.zeros(dim, dtype=np.float32)
        for t in tokens:
            if t in gene2idx:
                vec += embs[gene2idx[t]]
            else:
                missing.append(t)
        vectors.append(vec / len(tokens))

    pert_emb = np.stack(vectors).astype(np.float32)
    if missing:
        uniq_missing = sorted(set(missing))
        print(f"[gene2vec] missing genes ({len(uniq_missing)}), example: {uniq_missing[:1]}")
    print(f"[gene2vec] aligned: N={pert_emb.shape[0]}, dim={dim}")
    return pert_emb, dim


def main():
    args = create_argparser().parse_args()
    args.ctrl_dim = [int(i) for i in args.ctrl_dim.split(',')]
    args.pert_dim = [int(i) for i in args.pert_dim.split(',')]

    if args.mode == "ood" and not args.ood_data_dir:
        raise ValueError("--ood_data_dir is required when --mode=ood")

    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)

    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [
            m for m in os.listdir(args.multimodal_model_path)
            if m.startswith('model') and m.endswith('.pt')
            and int(m.split('.')[0][5:]) >= args.skip_steps
        ]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.multimodal_model_path, m) for m in multimodal_name_list]
    else:
        multimodal_name_list = args.multimodal_model_path.split(',')

    logger.log(f"models waiting to be evaluated:{multimodal_name_list}")

    for model_path in multimodal_name_list:
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"),
            is_strict=args.is_strict,
        )
        multimodal_model.to(dist_util.dev())
        if args.use_fp16:
            multimodal_model.convert_to_fp16()
        multimodal_model.eval()
        multimodal_model.specific_type = 0

        logger.log(f"sampling samples for {model_path} (mode={args.mode})")
        model_name = os.path.basename(model_path)
        out_root = os.path.join(args.output_dir, model_name, 'original')
        img_save_path = args.output_dir
        if dist_util.is_main_process():
            os.makedirs(out_root, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)

        adata_sc = sc.read(args.data_dir)
        adata_ctrl = adata_sc[adata_sc.obs["cp_type"] == "control"].copy()
        if args.mode == "ood":
            adata_pert = sc.read(args.ood_data_dir)
        else:
            adata_pert = adata_sc[adata_sc.obs["cp_type"] == "stimulated"].copy()

        classes_all = None
        if args.class_cond:
            labels = adata_pert.obs[args.condition].values
            classes_all = LabelEncoder().fit_transform(labels)

        with open(args.encoder_config, 'r') as f:
            autoencoder_args = yaml.safe_load(f)
        encoder_model = EncoderModel(
            in_dim=adata_ctrl.shape[1],
            n_cat=args.num_class if args.num_class is not None else 7,
            conditioning_covariate=args.condition,
            encoder_type='learnt_autoencoder',
            **autoencoder_args,
        )
        encoder_model.load_state_dict(
            torch.load(args.ae_path, map_location=dist_util.dev())["state_dict"]
        )
        encoder_model.eval()

        gene_emb, _ = build_perturbation_embeddings(
            adata_pert, args.gene2vec_path, perturbation_key=args.perturbation_key,
        )

        ctrl_X = adata_ctrl.X
        pert_X = adata_pert.X
        if issparse(ctrl_X):
            ctrl_X = ctrl_X.toarray()
        if issparse(pert_X):
            pert_X = pert_X.toarray()
        ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
        pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
        ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
        pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
        z_ctrl, _, _ = encoder_model.encode({"X_norm": ctrl_X_norm})
        z_pert, _, _ = encoder_model.encode({"X_norm": pert_X_norm})
        z_ctrl = z_ctrl.detach().cpu().numpy()
        z_pert = z_pert.detach().cpu().numpy()

        M = ot.dist(z_pert, z_ctrl, metric='euclidean')
        G = ot.emd(
            torch.ones(z_pert.shape[0]) / z_pert.shape[0],
            torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
            torch.tensor(M),
            numItermax=100000,
        )
        match_idx = torch.max(G, 1)[1].numpy()
        ctrl = z_ctrl[match_idx]

        npzfile = np.load(os.path.join('/'.join(args.ae_path.split('/')[:-2]), args.norm_factor_name))
        ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))

        audio_cond_all = np.expand_dims(ctrl / ctrl_std10, axis=1).astype(np.float32)

        output_name = "sample_data.npz" if args.mode == "ood" else "sample_data_alldata.npz"
        _run_ordered_sampling(
            args,
            multimodal_model,
            multimodal_diffusion,
            classes_all,
            audio_cond_all,
            gene_emb=gene_emb,
            img_save_path=img_save_path,
            output_name=output_name,
        )

    logger.log("sampling complete")


def _run_ordered_sampling(
    args,
    multimodal_model,
    multimodal_diffusion,
    classes_all,
    audio_cond_all,
    gene_emb,
    img_save_path,
    output_name,
):
    device = dist_util.dev()
    n = len(classes_all) if classes_all is not None else audio_cond_all.shape[0]
    out_audios = np.empty((n, *args.pert_dim), dtype=np.float32)
    out_labels = np.empty((n,), dtype=np.int64)
    out_ids = np.empty((n,), dtype=np.int64)

    bs = int(args.batch_size)
    num_chunks = math.ceil(n / bs)

    for chunk_id in range(num_chunks):
        start = chunk_id * bs
        end = min((chunk_id + 1) * bs, n)
        m = end - start
        idx = np.arange(start, end)

        classes_np = classes_all[idx] if classes_all is not None else np.zeros(m, dtype=np.int64)
        audio_cond_np = audio_cond_all[idx]

        model_kwargs = {
            "label": th.tensor(classes_np, device=device, dtype=th.int),
            "audio_cond": th.tensor(audio_cond_np, device=device, dtype=th.float32),
        }
        if gene_emb is not None:
            model_kwargs["audio_gene_cond"] = th.tensor(gene_emb[idx], device=device, dtype=th.float32)

        shape = {
            "video": (m, *args.ctrl_dim),
            "audio": (m, *args.pert_dim),
        }

        if args.sample_fn == 'dpm_solver':
            dpm_solver = multimodal_DPM_Solver(
                model=multimodal_model,
                alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32),
            )
            x_T = {
                "video": th.randn(shape["video"], device=device),
                "audio": th.randn(shape["audio"], device=device),
            }
            sample = dpm_solver.sample(x_T, steps=20, order=3, skip_type="logSNR", method="singlestep")
        elif args.sample_fn == 'dpm_solver++':
            dpm_solver = multimodal_DPM_Solver(
                model=multimodal_model,
                alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32),
                predict_x0=True, thresholding=True,
            )
            x_T = {
                "video": th.randn(shape["video"], device=device),
                "audio": th.randn(shape["audio"], device=device),
            }
            sample = dpm_solver.sample(x_T, steps=20, order=2, skip_type="logSNR", method="adaptive")
        else:
            sample_fn = (
                multimodal_diffusion.p_sample_loop
                if args.sample_fn == "ddpm"
                else multimodal_diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                multimodal_model,
                shape=shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        audio_np = sample["audio"].detach().cpu().numpy()
        out_audios[idx] = audio_np
        out_labels[idx] = classes_np.astype(np.int64)
        out_ids[idx] = idx

        if dist.is_initialized():
            dist.barrier()

    output_path = os.path.join(img_save_path, output_name)
    np.savez(output_path, pert=out_audios, label=out_labels, idx=out_ids)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        ref_path="",
        batch_size=16,
        sample_fn="dpm_solver",
        multimodal_model_path="",
        output_dir="",
        classifier_scale=0,
        devices=None,
        is_strict=True,
        all_save_num=1024,
        seed=42,
        load_noise="",
        data_dir="",
        condition='celltype',
        specific_type=None,
        skip_steps=0,
        encoder_config="path/to/encoder/default.yaml",
        ae_path="path/to/autoencoder.ckpt",
        gene2vec_path="path/to/gene2vec.txt",
        norm_factor_name="norm_factor.npz",
        perturbation_key="condition",
        mode="iid",
        ood_data_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print(th.cuda.current_device())
    main()
