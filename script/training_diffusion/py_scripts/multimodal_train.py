"""
Train a diffusion model on audio-video pairs.
"""
import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
from scduo.scduo_perturbation.diffusion import dist_util, logger
from scduo.scduo_perturbation.diffusion.multimodal_datasets import load_data_cell
from scduo.scduo_perturbation.diffusion.resample import create_named_schedule_sampler
from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from scduo.scduo_perturbation.diffusion.multimodal_train_util import TrainLoop
from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random
import torch


def load_training_data(args,dev="cuda:0"):
    data = load_data_cell(
        data_dir=args.data_dir,
        ae_path=args.ae_path,
        batch_size=args.batch_size,
        ctrl_dim=args.ctrl_dim,
        pert_dim=args.pert_dim,
        num_workers=args.num_workers,
        condition=args.condition,
        encoder_config=args.encoder_config,
        dev=dev,
    )
    
    # for audio_batch, class_num, pert_ctrl_cond in data:
    #     gt_batch = {"audio":audio_batch, "label":class_num, 
    #                 "audio_cond": pert_ctrl_cond}

    for audio_batch, class_num, pert_ctrl_cond, pert_gene_cond in data:
        gt_batch = {"audio":audio_batch, "label":class_num, 
                    "audio_cond": pert_ctrl_cond, "audio_gene_cond": pert_gene_cond}
      
        yield gt_batch



def main():
    args = create_argparser().parse_args()
    args.ctrl_dim = [int(i) for i in args.ctrl_dim.split(',')]
    args.pert_dim = [int(i) for i in args.pert_dim.split(',')]
    logger.configure(args.output_dir)
    dist_util.setup_dist(args.devices)
   
    args = set_seed_logger_random(args)

    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_training_data(args)

    logger.log("creating data loader done")
   
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        lr=args.lr,
        t_lr=args.t_lr,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_db=args.use_db,
        sample_fn=args.sample_fn,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.0,
        t_lr=1e-4,
        seed=42,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        num_workers=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        devices=None,
        save_interval=10000,
        output_dir="",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_db=False,
        sample_fn="dpm_solver",
        frame_gap=1,
        num_class=None, #22 #14
        condition='cell_type',
        encoder_config='encoder_multimodal',
        ae_path='/stor/lep/workspace/multi_diffusion/CFGen/project_folder/experiments/train_autoencoder_openproblem_multimodal_fixed/checkpoints/last.ckpt'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print(torch.cuda.current_device())
    main()
