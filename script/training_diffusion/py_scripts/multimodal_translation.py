# """
# Generate a large batch of video-audio pairs
# """
# import sys,os
# sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
# import argparse
# import os
# import numpy as np
# import torch as th
# import torch.distributed as dist
# from einops import rearrange, repeat
# import muon as mu
# import yaml
# import scanpy as sc
# from scduo.sc.diffusion import dist_util, logger
# from scduo.sc.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )

# from scduo.sc.diffusion.common import set_seed_logger_random, delete_pkl
# from scduo.sc.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
# from scduo.sc.vae.data.data_loader import RNAseqLoader
# from scduo.sc.vae.models.base.vae_model import EncoderModel
# from scipy.sparse import issparse
# import torch

# def main():
#     args = create_argparser().parse_args()
#     args.ctrl_dim = [int(i) for i in args.ctrl_dim.split(',')]
#     args.pert_dim = [int(i) for i in args.pert_dim.split(',')]
    
    
#     dist_util.setup_dist(args.devices)
#     logger.configure(args.output_dir)
#     args = set_seed_logger_random(args)


#     logger.log("creating model and diffusion...")
#     multimodal_model, multimodal_diffusion = create_model_and_diffusion(
#          **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
#     )

#     if os.path.isdir(args.multimodal_model_path):
#         multimodal_name_list = [model_name for model_name in os.listdir(args.multimodal_model_path) \
#             if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
#         multimodal_name_list.sort()
#         multimodal_name_list = [os.path.join(args.model_path, model_name) for model_name in multimodal_name_list[::1]]
#     else:
#         multimodal_name_list = [model_path for model_path in args.multimodal_model_path.split(',')]
        
#     logger.log(f"models waiting to be evaluated:{multimodal_name_list}")


#     sr_noise=None
#     if os.path.exists(args.load_noise):
#         sr_noise = np.load(args.load_noise)
#         sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
#         sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.ctrl_dim[0])
#         # if dist.get_rank()==0:
#         #单gpu
#         if dist_util.is_main_process():
#             logger.log(f"load noise form {args.load_noise}...")

#     for model_path in multimodal_name_list:
#         multimodal_model.load_state_dict_(
#             dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=args.is_strict
#         )
        
#         multimodal_model.to(dist_util.dev())
#         if args.use_fp16:
#             multimodal_model.convert_to_fp16()
#         multimodal_model.eval()

#         logger.log(f"sampling samples for {model_path}")
#         model_name = model_path.split('/')[-1]

#         groups= 0
#         multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
#         audio_save_path = os.path.join(args.output_dir)
#         img_save_path = os.path.join(args.output_dir)
#         #if dist.get_rank() == 0:
#         #单gpu
#         if dist_util.is_main_process():
#             os.makedirs(multimodal_save_path, exist_ok=True)
#             os.makedirs(audio_save_path, exist_ok=True)
#             os.makedirs(img_save_path, exist_ok=True)

#         NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
#         adata_sc = sc.read(NK_dataset_path)
#         adata_ctrl = adata_sc[adata_sc.obs["condition"] == "control"].copy()
#         adata_pert = adata_sc[adata_sc.obs["condition"] == "stimulated"].copy()
#         adata = {
#             "ctrl": adata_ctrl,
#             "pert": adata_pert
#         }
#         if args.class_cond:
#             from sklearn.preprocessing import LabelEncoder
#             labels = adata['ctrl'].obs[args.condition].values
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             # classes_all = label_encoder.transform(labels)
#             classes_all = label_encoder.transform(labels)
            
        
#         with open(args.encoder_config, 'r') as file:
#             yaml_content = file.read()
#         autoencoder_args = yaml.safe_load(yaml_content)

#         # Initialize encoder 
#         dataset = RNAseqLoader(
#             data_path=args.data_dir,
#             layer_key='X_counts',
#             covariate_keys=["cell_type"],
#             subsample_frac=1,
#             encoder_type='learnt_autoencoder',
#             condition_key="condition",
#             control_value="control",
#             perturbed_value="stimulated"
#         )            
#         gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}
#         encoder_model = EncoderModel(in_dim=gene_dim,
#                                             n_cat=6,
#                                             conditioning_covariate=args.condition, 
#                                             encoder_type='learnt_autoencoder',
#                                             **autoencoder_args)

#         # Load weights 
#         encoder_model.load_state_dict(th.load(args.ae_path)["state_dict"])
#         encoder_model.to(dist_util.dev())
#         # initialize the source modality
#         batch = {}

#         gt_ctrl = th.tensor(adata['ctrl'].X.toarray(),device=dist_util.dev())
#         gt_pert = th.tensor(adata['pert'].X.toarray(),device=dist_util.dev())
#         from scduo.sc.vae.data.utils import normalize_expression
#         gt_ctrl_norm = normalize_expression(gt_ctrl, gt_ctrl.sum(), encoder_type='learnt_autoencoder')
#         gt_pert_norm = normalize_expression(gt_pert, gt_pert.sum(), encoder_type='learnt_autoencoder')

#         batch["X_norm"] = {'ctrl':gt_ctrl_norm,'pert':gt_pert_norm}
#         z,_,_ = encoder_model.encode(batch)
#         noise_init = z[next(s for s in z.keys() if s != args.gen_mode)]

#         npzfile = np.load('/'.join(args.ae_path.split('/')[:-2])+'/norm_factor.npz1.npz')
#         std = npzfile['ctrl_std'] if args.gen_mode == 'pert' else npzfile['pert_std']
#         noise_init = noise_init/th.tensor(std,device=noise_init.device)


#         #注意力embbeding
#         dataset_path_1 = '/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad'
#         dataset_path_2 = '/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_oodNK.h5ad'
    
#         adata1 = sc.read(dataset_path_1)
#         adata2 = sc.read(dataset_path_2)
#         def get_lantent_adata(adata1,encoder_model):
#             adata_ctrl = adata1[adata1.obs["condition"] == "control"].copy()
#             adata_pert = adata1[adata1.obs["condition"] == "stimulated"].copy()

#             ctrl_X = adata_ctrl.X
#             pert_X = adata_pert.X

#             if issparse(ctrl_X):
#                 ctrl_X = ctrl_X.toarray()
#             if issparse(pert_X):
#                 pert_X = pert_X.toarray()

#             ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#             pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#             from scduo.sc.vae.data.utils import normalize_expression
#             ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#             pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')

#             batch = {}
#             batch["X_norm"] = {
#                 "ctrl" : ctrl_X_norm.clone().detach().to(dtype=encoder_model.dtype),
#                 "pert" : pert_X_norm.clone().detach().to(dtype=encoder_model.dtype),

#             }
#             z,_,_ = encoder_model.encode(batch)
#             # latent_adata = sc.AnnData(X=z["ctrl"], obs=adata_ctrl.obs.copy())
#             latent_adata = sc.AnnData(X=z["ctrl"].detach().cpu().numpy(), obs=adata_ctrl.obs.copy())
#             return latent_adata

#         test_z = get_lantent_adata(adata1,encoder_model).X
#         ctrl_z=get_lantent_adata(adata2,encoder_model)
#         ctrl_list = []
#         from sklearn.preprocessing import LabelEncoder
#         labels = adata2.obs["cell_type"].values
#         label_encoder = LabelEncoder()
#         label_encoder.fit(labels)
#         classes = label_encoder.transform(labels)
#         types = list(label_encoder.classes_)
#         for cell_type in types:
#                 ctrl_m = ctrl_z[(ctrl_z.obs["cell_type"] == cell_type)].to_df().values.mean(axis=0)
#                 if len(ctrl_list) > 0:
#                     ctrl_list = np.vstack((ctrl_list, ctrl_m))
#                 else:
#                     ctrl_list = ctrl_m

#         from sklearn.metrics.pairwise import cosine_similarity
#         from sklearn.preprocessing import normalize
#         cos_sim_all = cosine_similarity(np.array(test_z),
#                                             np.array(ctrl_list))
#         top_k = 2
#         filtered_cos_sim = np.zeros_like(cos_sim_all)
#         for i in range(cos_sim_all.shape[0]):
#             row = cos_sim_all[i]
#             # 获取 top_k 最大值的索引
#             top_idx = np.argpartition(row, -top_k)[-top_k:]
#             top_values = row[top_idx]

#             # 归一化
#             normed = top_values / top_values.sum()

#             # 将归一化后的 top_k 值写入对应位置
#             filtered_cos_sim[i, top_idx] = normed

#         # 用新的注意力矩阵替代
#         cos_sim_all = filtered_cos_sim
#         # cos_sim_all = normalize(cos_sim_all, axis=1, norm='l1')

#         videos = []
#         audios = []
#         all_labels = []
#         cos_sim_list=[]

#         # while groups * args.batch_size *  dist.get_world_size()< args.all_save_num: 
#         sample_num = noise_init.shape[0]
#         num_iteration = int(sample_num/args.batch_size)+1
#         for i in list(range(num_iteration))*args.gen_times:
       
#             model_kwargs = {}

#             start_idx = i * args.batch_size
#             end_idx = min((i + 1) * args.batch_size, cos_sim_all.shape[0])
#             cos_sim = cos_sim_all[start_idx:end_idx]
#             model_kwargs["cos_sim"] = torch.tensor(cos_sim, dtype=torch.float32, device=dist_util.dev())
            
#             x_T_init = noise_init[i*args.batch_size:(i+1)*args.batch_size]
#             if args.gen_mode == 'ctrl':
#                 model_kwargs["audio"] = x_T_init.unsqueeze(1).to(dist_util.dev())
#             else:
#                 model_kwargs["video"] = x_T_init.unsqueeze(1).to(dist_util.dev())
#             if args.class_cond:
#                 classes = classes_all[i*args.batch_size:(i+1)*args.batch_size]  # generated random cell type
#                 classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
#                 model_kwargs["label"] = classes

        
#             shape = {"video":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.ctrl_dim), \
#                     "audio":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.pert_dim)
#                 }
#             if args.sample_fn == 'dpm_solver':
#                 # sample_fn = multimodal_dpm_solver
#                 # sample = sample_fn(shape = shape, \
#                 #     model_fn = multimodal_model, steps=args.timestep_respacing)

#                 dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
#                     alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32))
#                 x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
#                         "audio":th.randn(shape["audio"]).to(dist_util.dev())}
#                 sample = dpm_solver.sample(
#                     x_T,
#                     steps=20,
#                     order=3,
#                     skip_type="logSNR",
#                     method="singlestep",
#                 )

#             elif args.sample_fn == 'dpm_solver++':
#                 dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
#                     alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32), \
#                         predict_x0=True, thresholding=True)
                
#                 x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
#                         "audio":th.randn(shape["audio"]).to(dist_util.dev())}
#                 sample = dpm_solver.sample(
#                     x_T,
#                     steps=20,
#                     order=2,
#                     skip_type="logSNR",
#                     method="adaptive",
#                 )
#             else:
#                 sample_fn = (
#                     multimodal_diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
#                 )

#                 sample = sample_fn(
#                     multimodal_model,
#                     shape = shape,
#                     clip_denoised=args.clip_denoised,
#                     model_kwargs=model_kwargs,
#                     # noise=x_T_init,
#                     # gen_mode=args.gen_mode,
#                     use_fp16 = args.use_fp16,
#                     class_scale=args.classifier_scale
#                 )

#             video = sample["video"]
#             audio = sample["audio"]  


#             all_videos = video.detach().cpu().numpy()
#             all_audios = audio.detach().cpu().numpy()

#             if args.class_cond:
#                 all_labels.append(classes.cpu().numpy())
                
#             videos.append(all_videos)
#             audios.append(all_audios)
#             cos_sim_list.append(cos_sim)

#             groups += 1

#             #单gpu
#             if dist.is_initialized():
#                 dist.barrier()



#         videos = np.concatenate(videos)
#         audios = np.concatenate(audios)
#         cos_sim_array = np.concatenate(cos_sim_list)

#         all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
               
#         save_path = os.path.join(args.output_dir, "translated_sample1.npz")

#         np.savez(save_path, ctrl=videos, pert=audios, label=all_labels,cos_sim=cos_sim_array)


#     logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         ref_path="",
#         batch_size=16,
#         sample_fn="dpm_solver",
#         multimodal_model_path="",
#         output_dir="",
#         classifier_scale=0.0,
#         devices=None,
#         is_strict=True,
#         all_save_num= 1024,
#         seed=42,
#         load_noise="",
#         data_dir="",
#         gen_mode='pert',
#         class_cond=True,
#         encoder_config='default',
#         condition='cell_type',
#         gen_times=5,
#         ae_path='/home/wuboyang/scduo-main/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v1.ckpt',
#     )
   
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# if __name__ == "__main__":
#     print(th.cuda.current_device())
#     main()










"""
Generate a large batch of video-audio pairs
"""
import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
import muon as mu
import yaml
import scanpy as sc
from scduo.sc.diffusion import dist_util, logger
from scduo.sc.diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)

from scduo.sc.diffusion.common import set_seed_logger_random, delete_pkl
from scduo.sc.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from scduo.sc.vae.data.data_loader import RNAseqLoader
from scduo.sc.vae.models.base.vae_model import EncoderModel
from scipy.sparse import issparse
import torch

def main():
    args = create_argparser().parse_args()
    args.ctrl_dim = [int(i) for i in args.ctrl_dim.split(',')]
    args.pert_dim = [int(i) for i in args.pert_dim.split(',')]
    
    
    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)


    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [model_name for model_name in os.listdir(args.multimodal_model_path) \
            if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.model_path, model_name) for model_name in multimodal_name_list[::1]]
    else:
        multimodal_name_list = [model_path for model_path in args.multimodal_model_path.split(',')]
        
    logger.log(f"models waiting to be evaluated:{multimodal_name_list}")


    sr_noise=None
    if os.path.exists(args.load_noise):
        sr_noise = np.load(args.load_noise)
        sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.ctrl_dim[0])
        # if dist.get_rank()==0:
        #单gpu
        if dist_util.is_main_process():
            logger.log(f"load noise form {args.load_noise}...")

    for model_path in multimodal_name_list:
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=args.is_strict
        )
        
        multimodal_model.to(dist_util.dev())
        if args.use_fp16:
            multimodal_model.convert_to_fp16()
        multimodal_model.eval()

        logger.log(f"sampling samples for {model_path}")
        model_name = model_path.split('/')[-1]

        groups= 0
        multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
        audio_save_path = os.path.join(args.output_dir)
        img_save_path = os.path.join(args.output_dir)
        #if dist.get_rank() == 0:
        #单gpu
        if dist_util.is_main_process():
            os.makedirs(multimodal_save_path, exist_ok=True)
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)

        dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
        adata_sc = sc.read(dataset_path)
        adata_ctrl = adata_sc[(adata_sc.obs['condition'] == 'control') & (adata_sc.obs['cell_type'] == 'NK')].copy()
        adata_pert = adata_sc[(adata_sc.obs['condition'] == 'stimulated') & (adata_sc.obs['cell_type'] == 'NK')].copy()
        adata = {
            "ctrl": adata_ctrl,
            "pert": adata_pert
        }
        if args.class_cond:
            from sklearn.preprocessing import LabelEncoder
            labels = adata_sc[(adata_sc.obs['condition'] == 'control')].obs[args.condition].values
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            classes_all = label_encoder.transform(labels)+3
            # cd4t_labels = adata_ctrl.obs[args.condition].values
            # classes_all = label_encoder.transform(cd4t_labels)
            
        
        with open(args.encoder_config, 'r') as file:
            yaml_content = file.read()
        autoencoder_args = yaml.safe_load(yaml_content)

        # Initialize encoder 
        dataset = RNAseqLoader(
            data_path=args.data_dir,
            layer_key='X_counts',
            covariate_keys=["cell_type"],
            subsample_frac=1,
            encoder_type='learnt_autoencoder',
            condition_key="condition",
            control_value="control",
            perturbed_value="stimulated"
        )            
        gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}
        encoder_model = EncoderModel(in_dim=gene_dim,
                                            n_cat=6,
                                            conditioning_covariate=args.condition, 
                                            encoder_type='learnt_autoencoder',
                                            **autoencoder_args)

        # Load weights 
        encoder_model.load_state_dict(th.load(args.ae_path)["state_dict"])
        encoder_model.to(dist_util.dev())

        # initialize the source modality
        batch = {}

        gt_ctrl = th.tensor(adata['ctrl'].X.toarray(),device=dist_util.dev())
        gt_pert = th.tensor(adata['pert'].X.toarray(),device=dist_util.dev())
        from scduo.sc.vae.data.utils import normalize_expression
        gt_ctrl_norm = normalize_expression(gt_ctrl, gt_ctrl.sum(), encoder_type='learnt_autoencoder')
        gt_pert_norm = normalize_expression(gt_pert, gt_pert.sum(), encoder_type='learnt_autoencoder')


        batch["X_norm"] = {'ctrl':gt_ctrl_norm,'pert':gt_pert_norm}
        z,_,_ = encoder_model.encode(batch)

        #扰动前后数据作为条件引导的潜在空间向量处理
        ctrl_std10 = z["ctrl"].std(0).mean()*10
        pert_std10 = z["pert"].std(0).mean()*10
        noise_init = z
        npzfile = np.load('/'.join(args.ae_path.split('/')[:-2])+'/norm_factor.npz2.npz')
        std = npzfile
        noise_init = {"ctrl": noise_init["ctrl"]/th.tensor(std["ctrl_std"],device=noise_init["ctrl"].device), \
                      "pert": noise_init["pert"]/th.tensor(std['pert_std'],device=noise_init["pert"].device)}
        pred_pert=np.load("/home/wuboyang/scduo-main/script/training_diffusion/outputs/checkpoints/my_dfbackbone/output6/predict_output/pred_z.npz")["pred_z"]
        pred_pert = th.tensor(pred_pert, device=dist_util.dev(), dtype=th.float32)
        pred_pert=pred_pert/th.tensor(std['pert_std'],device=pred_pert.device)

        #单数据作引导
        # noise_init = z[next(s for s in z.keys() if s != args.gen_mode)]
        # npzfile = np.load('/'.join(args.ae_path.split('/')[:-2])+'/norm_factor.npz2.npz')
        # std = npzfile['ctrl_std'] if args.gen_mode == 'pert' else npzfile['pert_std']
        # noise_init = noise_init/th.tensor(std,device=noise_init.device)


        videos = []
        audios = []
        all_labels = []
        cos_sim_list=[]

        # while groups * args.batch_size *  dist.get_world_size()< args.all_save_num: 
        sample_num = noise_init["ctrl"].shape[0]
        num_iteration = int(sample_num/args.batch_size)+1
        for i in list(range(num_iteration))*args.gen_times:
       
            model_kwargs = {}

            #单数据作引导
            # x_T_init = noise_init[i*args.batch_size:(i+1)*args.batch_size]
            # if args.gen_mode == 'ctrl':
            #     model_kwargs["audio"] = x_T_init.unsqueeze(1).to(dist_util.dev())
            # else:
            #     model_kwargs["video"] = x_T_init.unsqueeze(1).to(dist_util.dev())
            # if args.class_cond:
            #     classes = classes_all[i*args.batch_size:(i+1)*args.batch_size]  # generated random cell type
            #     classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
            #     model_kwargs["label"] = classes
            # shape = {"video":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.ctrl_dim), \
            #         "audio":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.pert_dim)
            #         }
                
            #扰动前后数据同时作为条件引导
            ctrl_init = noise_init["ctrl"][i*args.batch_size:(i+1)*args.batch_size]
            # pert_init = noise_init["pert"][i*args.batch_size:(i+1)*args.batch_size]
            pert_init = pred_pert[i*args.batch_size:(i+1)*args.batch_size]
            model_kwargs["audio"] = pert_init.unsqueeze(1).to(dist_util.dev())
            model_kwargs["video"] = ctrl_init.unsqueeze(1).to(dist_util.dev())
            if args.class_cond:
                classes = classes_all[i*args.batch_size:(i+1)*args.batch_size]  # generated random cell type
                classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
                model_kwargs["label"] = classes
            shape = {"video":(args.batch_size if i!=num_iteration-1 else ctrl_init.shape[0], *args.ctrl_dim), \
                    "audio":(args.batch_size if i!=num_iteration-1 else pert_init.shape[0], *args.pert_dim)
                }

            if args.sample_fn == 'dpm_solver':
                # sample_fn = multimodal_dpm_solver
                # sample = sample_fn(shape = shape, \
                #     model_fn = multimodal_model, steps=args.timestep_respacing)

                dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32))
                x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
                        "audio":th.randn(shape["audio"]).to(dist_util.dev())}
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=3,
                    skip_type="logSNR",
                    method="singlestep",
                )

            elif args.sample_fn == 'dpm_solver++':
                dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32), \
                        predict_x0=True, thresholding=True)
                
                x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
                        "audio":th.randn(shape["audio"]).to(dist_util.dev())}
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=2,
                    skip_type="logSNR",
                    method="adaptive",
                )
            else:
                sample_fn = (
                    multimodal_diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
                )

                sample = sample_fn(
                    multimodal_model,
                    shape = shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    # noise=x_T_init,
                    # gen_mode=args.gen_mode,
                    use_fp16 = args.use_fp16,
                    class_scale=args.classifier_scale
                )

            video = sample["video"]
            audio = sample["audio"]  


            all_videos = video.detach().cpu().numpy()
            all_audios = audio.detach().cpu().numpy()

            if args.class_cond:
                all_labels.append(classes.cpu().numpy())
                
            videos.append(all_videos)
            audios.append(all_audios)
            # cos_sim_list.append(cos_sim)

            groups += 1

            #单gpu
            if dist.is_initialized():
                dist.barrier()



        videos = np.concatenate(videos)
        audios = np.concatenate(audios)
        # cos_sim_array = np.concatenate(cos_sim_list)

        all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
               
        save_path = os.path.join(args.output_dir, "translated_sample1.npz")

        # np.savez(save_path, ctrl=videos, pert=audios, label=all_labels,cos_sim=cos_sim_array)
        np.savez(save_path, ctrl=videos, pert=audios, label=all_labels)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        ref_path="",
        batch_size=16,
        sample_fn="dpm_solver",
        multimodal_model_path="",
        output_dir="",
        classifier_scale=0.0,
        devices=None,
        is_strict=True,
        all_save_num= 1024,
        seed=42,
        load_noise="",
        data_dir="",
        gen_mode='pert',
        class_cond=True,
        encoder_config='default',
        condition='cell_type',
        gen_times=5,
        ae_path='/home/wuboyang/scduo-main/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v1.ckpt',
    )
   
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print(th.cuda.current_device())
    main()
