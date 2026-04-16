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

# from scduo.scduo_perturbation.diffusion import dist_util, logger
# from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )
# from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random, delete_pkl
# from scduo.scduo_perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
# import scanpy as sc
# from distutils.spawn import spawn
# import random
# import blobfile as bf
# import numpy as np
# import torch as th
# import os
# import pickle
# import torch as th
# from torch.utils.data import DataLoader, Dataset

# import scanpy as sc
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# import yaml

# import muon as mu
# from muon import MuData
# from scipy.sparse import issparse
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel

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
#         sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
#         multimodal_model.specific_type = 0

#         logger.log(f"sampling samples for {model_path}")
#         model_name = model_path.split('/')[-1]

#         groups= 0
#         multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
#         sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
#         audio_save_path = os.path.join(args.output_dir)
#         img_save_path = os.path.join(args.output_dir)
#         #if dist.get_rank() == 0:
#         #单gpu
#         if dist_util.is_main_process():
#             os.makedirs(multimodal_save_path, exist_ok=True)
#             os.makedirs(sr_save_path, exist_ok=True)
#             os.makedirs(audio_save_path, exist_ok=True)
#             os.makedirs(img_save_path, exist_ok=True)


#         # NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
#         adata_sc = sc.read(args.data_dir)
#         adata_ctrl = adata_sc[adata_sc.obs["condition"] == "control"].copy()
#         adata_pert = adata_sc[adata_sc.obs["condition"] == "stimulated"].copy()
#         adata = {
#             "ctrl": adata_ctrl,
#             "pert": adata_pert
#         }
#         if args.class_cond:
#             from sklearn.preprocessing import LabelEncoder
#             labels = adata['pert'].obs[args.condition].values
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             classes_all = label_encoder.transform(labels)

#         ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v2.ckpt"
#         encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
#         with open(encoder_config, 'r') as file:
#             yaml_content = file.read()
#         autoencoder_args = yaml.safe_load(yaml_content)

#         # Initialize encoder
#         encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
#                                             n_cat=7,
#                                             conditioning_covariate=args.condition, 
#                                             encoder_type='learnt_autoencoder',
#                                             **autoencoder_args)
        
#         # Load weights 
#         encoder_model.load_state_dict(torch.load(ae_path, map_location=dist_util.dev())["state_dict"])
#         encoder_model.eval()

#         ctrl = []
#         pert = []
#         bs = 1000

#         batch_num = int(adata_ctrl.shape[0]/bs)+1
#         for i in range(batch_num):
#             batch = {}
#             ctrl_X = adata_ctrl[i*bs:(i+1)*bs].X
#             pert_X = adata_pert[i*bs:(i+1)*bs].X

#             if issparse(ctrl_X):
#                 ctrl_X = ctrl_X.toarray()
#             if issparse(pert_X):
#                 pert_X = pert_X.toarray()
            
#             ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#             pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#             from scduo.scduo_perturbation.vae.data.utils import normalize_expression
#             ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#             pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
            
#             all_norm = torch.cat([ctrl_X_norm, pert_X_norm], dim=0)
#             batch["X_norm"] = all_norm
#             half=ctrl_X.shape[0]

#             z,_,_ = encoder_model.encode(batch)
#             ctrl.append(z[:half])
#             pert.append(z[half:])
        
#         # rescaling into std = 1
#         ctrl = torch.concat(ctrl).to(dtype=torch.float32, device=dist_util.dev())
#         pert = torch.concat(pert).to(dtype=torch.float32, device=dist_util.dev())
#         npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor3.npz')
#         ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
#         pert_std10 = float(np.asarray(npzfile["pert_std"]))
#         # ctrl_std10 = ctrl.std(0).mean()*10
#         # pert_std10 = pert.std(0).mean()*10

#         with torch.no_grad():
#             audio_cond_all = ((ctrl / ctrl_std10).unsqueeze(1)).cpu().numpy().astype(np.float32)
            


        
#         videos = []
#         audios = []
#         all_labels = []
#         ids = []
        
#         try:
#             world_size = dist.get_world_size()
#         except ValueError:
#             world_size = 1

#         while groups * args.batch_size *  world_size< args.all_save_num: 
#             model_kwargs = {}
#             if args.class_cond:
#                 if args.specific_type is None:
#                     n = len(classes_all)
#                     if args.batch_size > n:
#                         raise ValueError(f"batch_size({args.batch_size}) 不能大于类别总数({n})")
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     classes = classes_all[idx]
#                     audio_cond = audio_cond_all[idx]
#                     ids.append(idx)
                    
#                    #classes = np.random.choice(classes_all, args.batch_size, replace=False)  # generated random cell type
#                 else:
#                     print(f'generating {int(args.specific_type)} cell')
#                     n = ctrl.shape[0]
#                     classes = th.ones(args.batch_size)*int(args.specific_type)   # generated certain cell type
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     audio_cond = audio_cond_all[idx]
                    
#                 classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
#                 audio_cond = th.tensor(audio_cond, device=dist_util.dev(), dtype=th.float32)
                
#                 model_kwargs["label"] = classes
#                 model_kwargs["audio_cond"] = audio_cond
                

#             shape = {"video":(args.batch_size , *args.ctrl_dim), \
#                     "audio":(args.batch_size , *args.pert_dim)
#                 }
#             if args.sample_fn == 'dpm_solver':
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
#                     multimodal_diffusion.p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
#                 )

#                 sample = sample_fn(
#                     multimodal_model,
#                     shape = shape,
#                     clip_denoised=args.clip_denoised,
#                     model_kwargs=model_kwargs,
#                 )

#             audio = sample["audio"]              

#             all_audios = audio.detach().cpu().numpy()

#             if args.class_cond:
#                 all_labels.append(classes.cpu().numpy())
                
#             audios.append(all_audios)

#             groups += 1

#             #单gpu
#             if dist.is_initialized():
#                 dist.barrier()

#         audios = np.concatenate(audios)
#         all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
#         ids = np.concatenate(ids)

#         output_path = os.path.join(img_save_path, f"sample_data.npz")

#         np.savez(output_path,pert=audios,label=all_labels,idx = ids)


#     logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         ref_path="",
#         batch_size=16,
#         sample_fn="dpm_solver",
#         multimodal_model_path="",
#         output_dir="",
#         classifier_scale=0,
#         devices=None,
#         is_strict=True,
#         all_save_num= 1024,
#         seed=42,
#         load_noise="",
#         data_dir="",
#         condition='cell_type',
#         specific_type=None,
#     )
   
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# if __name__ == "__main__":
#     print(th.cuda.current_device())
#     main()

































# #单diffusion药物扰动--全数据集范围采样
# import sys,os
# sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
# import argparse
# import os
# import numpy as np
# import torch as th
# import torch.distributed as dist
# from einops import rearrange, repeat
# import muon as mu

# from scduo.scduo_perturbation.diffusion import dist_util, logger
# from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )
# from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random, delete_pkl
# from scduo.scduo_perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
# import scanpy as sc
# from distutils.spawn import spawn
# import random
# import blobfile as bf
# import numpy as np
# import torch as th
# import os
# import pickle
# import torch as th
# from torch.utils.data import DataLoader, Dataset

# import scanpy as sc
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# import yaml

# import muon as mu
# from muon import MuData
# from scipy.sparse import issparse
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
# import ot
# from torch.utils.data import DataLoader, TensorDataset

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
#         sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
#         multimodal_model.specific_type = 0

#         logger.log(f"sampling samples for {model_path}")
#         model_name = model_path.split('/')[-1]

#         groups= 0
#         multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
#         sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
#         audio_save_path = os.path.join(args.output_dir)
#         img_save_path = os.path.join(args.output_dir)
#         #if dist.get_rank() == 0:
#         #单gpu
#         if dist_util.is_main_process():
#             os.makedirs(multimodal_save_path, exist_ok=True)
#             os.makedirs(sr_save_path, exist_ok=True)
#             os.makedirs(audio_save_path, exist_ok=True)
#             os.makedirs(img_save_path, exist_ok=True)


#         # NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
#         adata_sc = sc.read(args.data_dir)
#         adata_ctrl = adata_sc[adata_sc.obs["condition"] == "control"].copy()
#         adata_pert = adata_sc[adata_sc.obs["condition"] == "stimulated"].copy()
#         adata = {
#             "ctrl": adata_ctrl,
#             "pert": adata_pert
#         }
#         if args.class_cond:
#             from sklearn.preprocessing import LabelEncoder
#             labels = adata['pert'].obs[args.condition].values
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             classes_all = label_encoder.transform(labels)

#         ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v17.ckpt"
#         encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
#         with open(encoder_config, 'r') as file:
#             yaml_content = file.read()
#         autoencoder_args = yaml.safe_load(yaml_content)

#         # Initialize encoder
#         encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
#                                             n_cat=7,
#                                             conditioning_covariate=args.condition, 
#                                             encoder_type='learnt_autoencoder',
#                                             **autoencoder_args)
        
#         # Load weights 
#         encoder_model.load_state_dict(torch.load(ae_path, map_location=dist_util.dev())["state_dict"])
#         encoder_model.eval()

#         ctrl_X = adata_ctrl.X
#         pert_X = adata_pert.X
#         if issparse(ctrl_X):
#             ctrl_X = ctrl_X.toarray()
#         if issparse(pert_X):
#             pert_X = pert_X.toarray()
#         ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#         pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#         from scduo.scduo_perturbation.vae.data.utils import normalize_expression
#         ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#         pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
#         z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
#         z_pert,_,_ = encoder_model.encode({"X_norm":pert_X_norm})
#         z_ctrl = z_ctrl.detach().cpu().numpy()
#         z_pert = z_pert.detach().cpu().numpy()
#         M = ot.dist(z_pert, z_ctrl, metric='euclidean')
#         G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
#                     torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
#                     torch.tensor(M), numItermax=100000)
#         match_idx = torch.max(G, 1)[1].numpy()
#         ctrl_new = z_ctrl[match_idx]


#         ctrl = ctrl_new
#         pert = z_pert
        
#         npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor18.npz')
#         ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
#         pert_std10 = float(np.asarray(npzfile["pert_std"]))

#         audio_cond_all = np.expand_dims(ctrl / ctrl_std10, axis=1).astype(np.float32)
            


        
#         videos = []
#         audios = []
#         all_labels = []
#         ids = []
        
#         try:
#             world_size = dist.get_world_size()
#         except ValueError:
#             world_size = 1

#         #分批次采样    
#         n = len(classes_all)
#         batch_size = args.batch_size  

#         classes = th.tensor(classes_all, device=dist_util.dev(), dtype=th.int)
#         audio_cond = th.tensor(audio_cond_all, device=dist_util.dev(), dtype=th.float32)

#         dataset = TensorDataset(classes, audio_cond)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#         # 采样的 main loop
#         audios = []
#         all_labels = []
#         ids = np.arange(n)

#         for i, (class_batch, audio_cond_batch) in enumerate(dataloader):
#             model_kwargs = {}
            
#             # 设置class_cond 和 audio_cond
#             model_kwargs["label"] = class_batch
#             model_kwargs["audio_cond"] = audio_cond_batch
#             batch = class_batch.shape[0]
#             shape = {"video": (batch, *args.ctrl_dim), "audio": (batch, *args.pert_dim)}
            
#             # 选择采样方法
#             if args.sample_fn == 'dpm_solver':
#                 dpm_solver = multimodal_DPM_Solver(model=multimodal_model,
#                                                 alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32))
#                 x_T = {"video": th.randn(shape["video"]).to(dist_util.dev()),
#                     "audio": th.randn(shape["audio"]).to(dist_util.dev())}
#                 sample = dpm_solver.sample(
#                     x_T,
#                     steps=20,
#                     order=3,
#                     skip_type="logSNR",
#                     method="singlestep",
#                 )
            
#             elif args.sample_fn == 'dpm_solver++':
#                 dpm_solver = multimodal_DPM_Solver(model=multimodal_model,
#                                                 alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32),
#                                                 predict_x0=True, thresholding=True)
#                 x_T = {"video": th.randn(shape["video"]).to(dist_util.dev()),
#                     "audio": th.randn(shape["audio"]).to(dist_util.dev())}
#                 sample = dpm_solver.sample(
#                     x_T,
#                     steps=20,
#                     order=2,
#                     skip_type="logSNR",
#                     method="adaptive",
#                 )
#             else:
#                 sample_fn = (multimodal_diffusion.p_sample_loop if args.sample_fn == "ddpm" else multimodal_diffusion.ddim_sample_loop)
#                 sample = sample_fn(
#                     multimodal_model,
#                     shape=shape,
#                     clip_denoised=args.clip_denoised,
#                     model_kwargs=model_kwargs,
#                 )
            
#             # 提取音频
#             audio = sample["audio"]
            
#             # 存储采样结果
#             all_audios = audio.detach().cpu().numpy()
            
#             if args.class_cond:
#                 all_labels.append(class_batch.cpu().numpy())
            
#             audios.append(all_audios)
            
#             # 每个批次完成后，确保顺序采样
#             if dist.is_initialized():
#                 dist.barrier()

#         # 将所有音频和标签合并
#         audios = np.concatenate(audios)
#         all_labels = np.concatenate(all_labels) if all_labels else np.zeros(audios.shape[0])

#         # 保存结果
#         output_path = os.path.join(img_save_path, f"sample_data_alldata.npz")
#         np.savez(output_path, pert=audios, label=all_labels, idx=ids)

#         logger.log("sampling complete")


#     #     model_kwargs = {}
#     #     if args.class_cond:
#     #         n = len(classes_all)
#     #         idx = np.arange(n)
#     #         classes = classes_all[idx]
#     #         audio_cond = audio_cond_all[idx]
#     #         ids.append(idx)
            
#     #         #classes = np.random.choice(classes_all, args.batch_size, replace=False)  # generated random cell type
                
#     #         classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
#     #         audio_cond = th.tensor(audio_cond, device=dist_util.dev(), dtype=th.float32)
            
#     #         model_kwargs["label"] = classes
#     #         model_kwargs["audio_cond"] = audio_cond
            

#     #     shape = {"video":(n , *args.ctrl_dim), \
#     #             "audio":(n , *args.pert_dim)
#     #         }
#     #     if args.sample_fn == 'dpm_solver':
#     #         dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
#     #             alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32))
#     #         x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
#     #                 "audio":th.randn(shape["audio"]).to(dist_util.dev())}
#     #         sample = dpm_solver.sample(
#     #             x_T,
#     #             steps=20,
#     #             order=3,
#     #             skip_type="logSNR",
#     #             method="singlestep",
#     #         )

#     #     elif args.sample_fn == 'dpm_solver++':
#     #         dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
#     #             alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32), \
#     #                 predict_x0=True, thresholding=True)
            
#     #         x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
#     #                 "audio":th.randn(shape["audio"]).to(dist_util.dev())}
#     #         sample = dpm_solver.sample(
#     #             x_T,
#     #             steps=20,
#     #             order=2,
#     #             skip_type="logSNR",
#     #             method="adaptive",
#     #         )
#     #     else:
#     #         sample_fn = (
#     #             multimodal_diffusion.p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
#     #         )

#     #         sample = sample_fn(
#     #             multimodal_model,
#     #             shape = shape,
#     #             clip_denoised=args.clip_denoised,
#     #             model_kwargs=model_kwargs,
#     #         )

#     #     audio = sample["audio"]              

#     #     all_audios = audio.detach().cpu().numpy()

#     #     if args.class_cond:
#     #         all_labels.append(classes.cpu().numpy())
            
#     #     audios.append(all_audios)

#     #     #单gpu
#     #     if dist.is_initialized():
#     #         dist.barrier()

#     #     audios = np.concatenate(audios)
#     #     all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
#     #     ids = np.concatenate(ids)

#     #     output_path = os.path.join(img_save_path, f"sample_data_alldata.npz")

#     #     np.savez(output_path,pert=audios,label=all_labels,idx = ids)


#     # logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         ref_path="",
#         batch_size=16,
#         sample_fn="dpm_solver",
#         multimodal_model_path="",
#         output_dir="",
#         classifier_scale=0,
#         devices=None,
#         is_strict=True,
#         all_save_num= 1024,
#         seed=42,
#         load_noise="",
#         data_dir="",
#         condition='cell_type',
#         specific_type=None,
#     )
   
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# if __name__ == "__main__":
#     print(th.cuda.current_device())
#     main()











































# #ood实验
# import sys,os
# sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
# import argparse
# import os
# import numpy as np
# import torch as th
# import torch.distributed as dist
# from einops import rearrange, repeat
# import muon as mu

# from scduo.scduo_perturbation.diffusion import dist_util, logger
# from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )
# from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random, delete_pkl
# from scduo.scduo_perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
# import scanpy as sc
# from distutils.spawn import spawn
# import random
# import blobfile as bf
# import numpy as np
# import torch as th
# import os
# import pickle
# import torch as th
# from torch.utils.data import DataLoader, Dataset

# import scanpy as sc
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# import yaml

# import muon as mu
# from muon import MuData
# from scipy.sparse import issparse
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel

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
#         sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
#         multimodal_model.specific_type = 0

#         logger.log(f"sampling samples for {model_path}")
#         model_name = model_path.split('/')[-1]

#         groups= 0
#         multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
#         sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
#         audio_save_path = os.path.join(args.output_dir)
#         img_save_path = os.path.join(args.output_dir)
#         #if dist.get_rank() == 0:
#         #单gpu
#         if dist_util.is_main_process():
#             os.makedirs(multimodal_save_path, exist_ok=True)
#             os.makedirs(sr_save_path, exist_ok=True)
#             os.makedirs(audio_save_path, exist_ok=True)
#             os.makedirs(img_save_path, exist_ok=True)


#         # NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
#         adata_sc = sc.read(args.data_dir)
#         adata_ctrl = adata_sc[adata_sc.obs["condition"] == "control"].copy()
#         adata_pert = adata_sc[adata_sc.obs["condition"] == "stimulated"].copy()
#         adata = {
#             "ctrl": adata_ctrl,
#             "pert": adata_pert
#         }
#         if args.class_cond:
#             from sklearn.preprocessing import LabelEncoder
#             labels = adata['pert'].obs[args.condition].values
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             classes_all = label_encoder.transform(labels)

#         ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v2.ckpt"
#         encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
#         with open(encoder_config, 'r') as file:
#             yaml_content = file.read()
#         autoencoder_args = yaml.safe_load(yaml_content)

#         # Initialize encoder
#         encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
#                                             n_cat=7,
#                                             conditioning_covariate=args.condition, 
#                                             encoder_type='learnt_autoencoder',
#                                             **autoencoder_args)
        
#         # Load weights 
#         encoder_model.load_state_dict(torch.load(ae_path, map_location=dist_util.dev())["state_dict"])
#         encoder_model.eval()

#         ctrl = []
#         pert = []
#         bs = 1000

#         batch_num = int(adata_ctrl.shape[0]/bs)+1
#         for i in range(batch_num):
#             batch = {}
#             ctrl_X = adata_ctrl[i*bs:(i+1)*bs].X
#             pert_X = adata_pert[i*bs:(i+1)*bs].X

#             if issparse(ctrl_X):
#                 ctrl_X = ctrl_X.toarray()
#             if issparse(pert_X):
#                 pert_X = pert_X.toarray()
            
#             ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#             pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#             from scduo.scduo_perturbation.vae.data.utils import normalize_expression
#             ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#             pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
            
#             all_norm = torch.cat([ctrl_X_norm, pert_X_norm], dim=0)
#             batch["X_norm"] = all_norm
#             half=ctrl_X.shape[0]

#             z,_,_ = encoder_model.encode(batch)
#             ctrl.append(z[:half])
#             pert.append(z[half:])
        
#         # rescaling into std = 1
#         ctrl = torch.concat(ctrl).to(dtype=torch.float32, device=dist_util.dev())
#         pert = torch.concat(pert).to(dtype=torch.float32, device=dist_util.dev())
#         npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor1.npz')
#         ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
#         pert_std10 = float(np.asarray(npzfile["pert_std"]))
#         # ctrl_std10 = ctrl.std(0).mean()*10
#         # pert_std10 = pert.std(0).mean()*10

#         with torch.no_grad():
#             audio_cond_all = ((ctrl / ctrl_std10).unsqueeze(1)).cpu().numpy().astype(np.float32)
            


        
#         videos = []
#         audios = []
#         all_labels = []
#         ids = []
        
#         try:
#             world_size = dist.get_world_size()
#         except ValueError:
#             world_size = 1

#         while groups * args.batch_size *  world_size< args.all_save_num: 
#             model_kwargs = {}
#             if args.class_cond:
#                 if args.specific_type is None:
#                     n = len(classes_all)
#                     if args.batch_size > n:
#                         raise ValueError(f"batch_size({args.batch_size}) 不能大于类别总数({n})")
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     classes = classes_all[idx]
#                     audio_cond = audio_cond_all[idx]
                    
#                    #classes = np.random.choice(classes_all, args.batch_size, replace=False)  # generated random cell type
#                 else:
#                     print(f'generating {int(args.specific_type)} cell')
#                     n = ctrl.shape[0]
#                     classes = th.ones(args.batch_size)*int(args.specific_type)   # generated certain cell type
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     audio_cond = audio_cond_all[idx]
#                     ids.append(idx)
                    
#                 classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
#                 audio_cond = th.tensor(audio_cond, device=dist_util.dev(), dtype=th.float32)
                
#                 model_kwargs["label"] = classes
#                 model_kwargs["audio_cond"] = audio_cond
                

#             shape = {"video":(args.batch_size , *args.ctrl_dim), \
#                     "audio":(args.batch_size , *args.pert_dim)
#                 }
#             if args.sample_fn == 'dpm_solver':
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
#                     multimodal_diffusion.p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
#                 )

#                 sample = sample_fn(
#                     multimodal_model,
#                     shape = shape,
#                     clip_denoised=args.clip_denoised,
#                     model_kwargs=model_kwargs,
#                 )

#             audio = sample["audio"]              

#             all_audios = audio.detach().cpu().numpy()

#             if args.class_cond:
#                 all_labels.append(classes.cpu().numpy())
                
#             audios.append(all_audios)

#             groups += 1

#             #单gpu
#             if dist.is_initialized():
#                 dist.barrier()

#         audios = np.concatenate(audios)
#         all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
#         ids = np.concatenate(ids)
#         output_path = os.path.join(img_save_path, f"sample_data1.npz")

#         np.savez(output_path,pert=audios,label=all_labels,idx = ids)


#     logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         ref_path="",
#         batch_size=16,
#         sample_fn="dpm_solver",
#         multimodal_model_path="",
#         output_dir="",
#         classifier_scale=0,
#         devices=None,
#         is_strict=True,
#         all_save_num= 1024,
#         seed=42,
#         load_noise="",
#         data_dir="",
#         condition='cell_type',
#         specific_type=None,
#     )
   
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# if __name__ == "__main__":
#     print(th.cuda.current_device())
#     main()



















#单diffusion基因扰动实验--全数据集范围采样
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

from scduo.scduo_perturbation.diffusion import dist_util, logger
from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random, delete_pkl
from scduo.scduo_perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
import scanpy as sc
from distutils.spawn import spawn
import random
import blobfile as bf
import numpy as np
import torch as th
import os
import pickle
import torch as th
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import yaml

import muon as mu
from muon import MuData
from scipy.sparse import issparse
from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
import ot
import math

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
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
        multimodal_model.specific_type = 0

        logger.log(f"sampling samples for {model_path}")
        model_name = model_path.split('/')[-1]

        groups= 0
        multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
        sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
        audio_save_path = os.path.join(args.output_dir)
        img_save_path = os.path.join(args.output_dir)
        #if dist.get_rank() == 0:
        #单gpu
        if dist_util.is_main_process():
            os.makedirs(multimodal_save_path, exist_ok=True) 
            os.makedirs(sr_save_path, exist_ok=True)
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)


        # NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
        adata_sc = sc.read(args.data_dir)
        adata_ctrl = adata_sc[adata_sc.obs["cp_type"] == "control"].copy()
        adata_pert = adata_sc[adata_sc.obs["cp_type"] == "stimulated"].copy()
        adata = {
            "ctrl": adata_ctrl,
            "pert": adata_pert
        }
        if args.class_cond:
            from sklearn.preprocessing import LabelEncoder
            labels = adata['pert'].obs[args.condition].values
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            classes_all = label_encoder.transform(labels)

        ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v18.ckpt"
        encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
        gene2vec_path="/home/wuboyang/scduo-new/gene2vec/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
        with open(encoder_config, 'r') as file:
            yaml_content = file.read()
        autoencoder_args = yaml.safe_load(yaml_content)

        # Initialize encoder
        encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
                                            n_cat=7,
                                            conditioning_covariate=args.condition, 
                                            encoder_type='learnt_autoencoder',
                                            **autoencoder_args)
        
        # Load weights 
        encoder_model.load_state_dict(torch.load(ae_path, map_location=dist_util.dev())["state_dict"])
        encoder_model.eval()

        gene_emb, self_gene_emb_dim = build_perturbation_embeddings(adata_pert, gene2vec_path)


        ctrl_X = adata_ctrl.X
        pert_X = adata_pert.X
        if issparse(ctrl_X):
            ctrl_X = ctrl_X.toarray()
        if issparse(pert_X):
            pert_X = pert_X.toarray()
        ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
        pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
        from scduo.scduo_perturbation.vae.data.utils import normalize_expression
        ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
        pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
        z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
        z_pert,_,_ = encoder_model.encode({"X_norm":pert_X_norm})
        z_ctrl = z_ctrl.detach().cpu().numpy()
        z_pert = z_pert.detach().cpu().numpy()
        M = ot.dist(z_pert, z_ctrl, metric='euclidean')
        G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
                    torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
                    torch.tensor(M), numItermax=100000)
        match_idx = torch.max(G, 1)[1].numpy()
        ctrl_new = z_ctrl[match_idx]

        ctrl = ctrl_new
        pert = z_pert
        
        npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor19.npz')
        ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
        pert_std10 = float(np.asarray(npzfile["pert_std"]))

        audio_cond_all = np.expand_dims(ctrl / ctrl_std10, axis=1).astype(np.float32)
            
        
        videos = []
        audios = []
        all_labels = []
        ids = []
        #target_idx = np.where(adata_pert.obs["perturbation"] == "CEBPE_RUNX1T1")[0]


        try:
            world_size = dist.get_world_size()
        except ValueError:
            world_size = 1

        model_kwargs = {}
        device = dist_util.dev()
        # 1) 全量大小 n（保持原来的顺序，不再一次性喂给采样器）
        n = len(classes_all)

        # 2) 预分配输出（保证结果按原顺序写回）
        out_audios = np.empty((n, *args.pert_dim), dtype=np.float32)
        out_labels = np.empty((n,), dtype=np.int64)
        out_ids    = np.empty((n,), dtype=np.int64)

        # 3) 顺序分块遍历：每块大小 = args.batch_size
        bs = int(args.batch_size)
        num_chunks = math.ceil(n / bs)

        for chunk_id in range(num_chunks):
            start = chunk_id * bs
            end   = min((chunk_id + 1) * bs, n)
            m     = end - start                      # 当前块的实际 batch 大小

            # 这一块的顺序索引（保证“按顺序分开采样”）
            idx = np.arange(start, end)

            # 组装当前块的条件
            classes_np       = classes_all[idx]              # 保持原顺序
            audio_cond_np    = audio_cond_all[idx]
            audio_gene_np    = gene_emb[idx]

            classes          = th.tensor(classes_np, device=device, dtype=th.int)
            audio_cond       = th.tensor(audio_cond_np, device=device, dtype=th.float32)
            audio_gene_cond  = th.tensor(audio_gene_np, device=device, dtype=th.float32)

            model_kwargs = {
                "label": classes,
                "audio_cond": audio_cond,
                "audio_gene_cond": audio_gene_cond,
            }

            # 注意：shape 用当前块大小 m，而不是 n
            shape = {
                "video": (m, *args.ctrl_dim),
                "audio": (m, *args.pert_dim),
            }

            # ====== 采样 ======
            if args.sample_fn == 'dpm_solver':
                dpm_solver = multimodal_DPM_Solver(
                    model=multimodal_model,
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32)
                )
                x_T = {
                    "video": th.randn(shape["video"], device=device),
                    "audio": th.randn(shape["audio"], device=device),
                }
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=3,
                    skip_type="logSNR",
                    method="singlestep",
                )
            elif args.sample_fn == 'dpm_solver++':
                dpm_solver = multimodal_DPM_Solver(
                    model=multimodal_model,
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32),
                    predict_x0=True, thresholding=True
                )
                x_T = {
                    "video": th.randn(shape["video"], device=device),
                    "audio": th.randn(shape["audio"], device=device),
                }
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=2,
                    skip_type="logSNR",
                    method="adaptive",
                )
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

            # 取当前块的结果
            audio = sample["audio"]                       # (m, *args.pert_dim)
            audio_np = audio.detach().cpu().numpy()

            # ====== 写回到全局输出的对应位置（保证顺序）======
            out_audios[idx] = audio_np
            out_labels[idx] = classes_np.astype(np.int64)
            out_ids[idx]    = idx                         # 如果你有自定义 ids，这里替换为你的 ids 来源

            if dist.is_initialized():
                dist.barrier()

        # 4) 块循环结束后，再一次性保存（顺序已保证）
        audios      = out_audios
        all_labels  = out_labels
        ids         = out_ids

        output_path = os.path.join(img_save_path, "sample_data_alldata.npz")
        np.savez(output_path, pert=audios, label=all_labels, idx=ids)


    logger.log("sampling complete")


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
        all_save_num= 1024,
        seed=42,
        load_noise="",
        data_dir="",
        condition='celltype',
        specific_type=None,
    )
   
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


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



# def build_perturbation_embeddings(adata_pert,gene2vec_path):
#         genes, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
#         dim = embs.shape[1]
#         pert_series = adata_pert.obs["perturbation"].astype(str)
#         nperts_value = adata_pert.obs["nperts"].astype(int)

#         vectors = []
#         missing = []
#         for g, n in zip(pert_series, nperts_value):
#             if n == 1:
#                 if g in gene2idx:
#                     vectors.append(embs[gene2idx[g]])
#                 else:
#                     # 缺失时给 0；也可以改成小随机噪声
#                     vectors.append(np.zeros(dim, dtype=np.float32))
#                     missing.append(g)
#             elif n == 2:
#                 g1, g2 = g.split("_")
#                 vec = np.zeros(dim, dtype=np.float32)
#                 if g1 in gene2idx:
#                     vec += embs[gene2idx[g1]]
#                 else:
#                     missing.append(g1)
#                 if g2 in gene2idx:
#                     vec += embs[gene2idx[g2]]
#                 else:
#                     missing.append(g2)
#                 vectors.append(vec / 2)
#             else:
#                 print(f"[gene2vec] Warning: 发现 nperts > 2 的扰动 {g}")
                
#         pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
#         pert_emb_dim = dim
#         uniq_missing = sorted(set(missing))
#         if uniq_missing:
#             print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
#         print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
#         return pert_emb, pert_emb_dim
def build_perturbation_embeddings(adata_pert,gene2vec_path):
    genes, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
    dim = embs.shape[1]
    pert_series = adata_pert.obs["condition"].astype(str)

    vectors = []
    missing = []
    for g in pert_series:
        if g in gene2idx:
            vectors.append(embs[gene2idx[g]])
        else:
            # 缺失时给 0；也可以改成小随机噪声
            vectors.append(np.zeros(dim, dtype=np.float32))
            missing.append(g)

    pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
    pert_emb_dim = dim
    uniq_missing = sorted(set(missing))
    if uniq_missing:
        print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
    print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
    return pert_emb, pert_emb_dim




if __name__ == "__main__":
    print(th.cuda.current_device())
    main()


























# #单diffusion基因扰动实验
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

# from scduo.scduo_perturbation.diffusion import dist_util, logger
# from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )
# from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random, delete_pkl
# from scduo.scduo_perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
# import scanpy as sc
# from distutils.spawn import spawn
# import random
# import blobfile as bf
# import numpy as np
# import torch as th
# import os
# import pickle
# import torch as th
# from torch.utils.data import DataLoader, Dataset

# import scanpy as sc
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# import yaml

# import muon as mu
# from muon import MuData
# from scipy.sparse import issparse
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
# import ot

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
#         sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
#         multimodal_model.specific_type = 0

#         logger.log(f"sampling samples for {model_path}")
#         model_name = model_path.split('/')[-1]

#         groups= 0
#         multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
#         sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
#         audio_save_path = os.path.join(args.output_dir)
#         img_save_path = os.path.join(args.output_dir)
#         #if dist.get_rank() == 0:
#         #单gpu
#         if dist_util.is_main_process():
#             os.makedirs(multimodal_save_path, exist_ok=True) 
#             os.makedirs(sr_save_path, exist_ok=True)
#             os.makedirs(audio_save_path, exist_ok=True)
#             os.makedirs(img_save_path, exist_ok=True)


#         # NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
#         adata_sc = sc.read(args.data_dir)
#         adata_ctrl = adata_sc[adata_sc.obs["cp_type"] == "control"].copy()
#         adata_pert = adata_sc[adata_sc.obs["cp_type"] == "stimulated"].copy()
#         adata = {
#             "ctrl": adata_ctrl,
#             "pert": adata_pert
#         }
#         if args.class_cond:
#             from sklearn.preprocessing import LabelEncoder
#             labels = adata['pert'].obs[args.condition].values
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             classes_all = label_encoder.transform(labels)

#         ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v12.ckpt"
#         encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
#         gene2vec_path="/home/wuboyang/scduo-new/gene2vec/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
#         with open(encoder_config, 'r') as file:
#             yaml_content = file.read()
#         autoencoder_args = yaml.safe_load(yaml_content)

#         # Initialize encoder
#         encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
#                                             n_cat=7,
#                                             conditioning_covariate=args.condition, 
#                                             encoder_type='learnt_autoencoder',
#                                             **autoencoder_args)
        
#         # Load weights 
#         encoder_model.load_state_dict(torch.load(ae_path, map_location=dist_util.dev())["state_dict"])
#         encoder_model.eval()

#         gene_emb, self_gene_emb_dim = build_perturbation_embeddings(adata_pert, gene2vec_path)


#         ctrl_X = adata_ctrl.X
#         pert_X = adata_pert.X
#         if issparse(ctrl_X):
#             ctrl_X = ctrl_X.toarray()
#         if issparse(pert_X):
#             pert_X = pert_X.toarray()
#         ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#         pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#         from scduo.scduo_perturbation.vae.data.utils import normalize_expression
#         ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#         pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
#         z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
#         z_pert,_,_ = encoder_model.encode({"X_norm":pert_X_norm})
#         z_ctrl = z_ctrl.detach().cpu().numpy()
#         z_pert = z_pert.detach().cpu().numpy()
#         M = ot.dist(z_pert, z_ctrl, metric='euclidean')
#         G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
#                     torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
#                     torch.tensor(M), numItermax=100000)
#         match_idx = torch.max(G, 1)[1].numpy()
#         ctrl_new = z_ctrl[match_idx]

#         ctrl = ctrl_new
#         pert = z_pert
        
#         npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor13.npz')
#         ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
#         pert_std10 = float(np.asarray(npzfile["pert_std"]))

#         audio_cond_all = np.expand_dims(ctrl / ctrl_std10, axis=1).astype(np.float32)
            
        
#         videos = []
#         audios = []
#         all_labels = []
#         ids = []
#         target_idx = np.where(adata_pert.obs["perturbation"] == "CEBPE_RUNX1T1")[0]


#         try:
#             world_size = dist.get_world_size()
#         except ValueError:
#             world_size = 1

#         while groups * args.batch_size *  world_size< args.all_save_num: 
#             model_kwargs = {}
#             if args.class_cond:
#                 if args.specific_type is None:
#                     #n = len(classes_all)
#                     n = len(target_idx)
#                     if args.batch_size > n:
#                         raise ValueError(f"batch_size({args.batch_size}) 不能大于类别总数({n})")
#                     #idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     idx = np.random.choice(target_idx, size=args.batch_size, replace=False)
#                     classes = classes_all[idx]
#                     audio_cond = audio_cond_all[idx]
#                     audio_gene_cond = gene_emb[idx]
#                     ids.append(idx)
                    
#                    #classes = np.random.choice(classes_all, args.batch_size, replace=False)  # generated random cell type
#                 else:
#                     print(f'generating {int(args.specific_type)} cell')
#                     n = ctrl.shape[0]
#                     classes = th.ones(args.batch_size)*int(args.specific_type)   # generated certain cell type
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     audio_cond = audio_cond_all[idx]
                    
#                 classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
#                 audio_cond = th.tensor(audio_cond, device=dist_util.dev(), dtype=th.float32)
#                 audio_gene_cond = th.tensor(audio_gene_cond, device=dist_util.dev(), dtype=th.float32)

#                 model_kwargs["label"] = classes
#                 model_kwargs["audio_cond"] = audio_cond
#                 model_kwargs["audio_gene_cond"] = audio_gene_cond
                

#             shape = {"video":(args.batch_size , *args.ctrl_dim), \
#                     "audio":(args.batch_size , *args.pert_dim)
#                 }
#             if args.sample_fn == 'dpm_solver':
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
#                     multimodal_diffusion.p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
#                 )

#                 sample = sample_fn(
#                     multimodal_model,
#                     shape = shape,
#                     clip_denoised=args.clip_denoised,
#                     model_kwargs=model_kwargs,
#                 )

#             audio = sample["audio"]              

#             all_audios = audio.detach().cpu().numpy()

#             if args.class_cond:
#                 all_labels.append(classes.cpu().numpy())
                
#             audios.append(all_audios)

#             groups += 1

#             #单gpu
#             if dist.is_initialized():
#                 dist.barrier()

#         audios = np.concatenate(audios)
#         all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
#         ids = np.concatenate(ids)       
#         output_path = os.path.join(img_save_path, f"sample_data_CEBPE_RUNX1T1.npz")

#         np.savez(output_path,pert=audios,label=all_labels,idx=ids)


#     logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         ref_path="",
#         batch_size=16,
#         sample_fn="dpm_solver",
#         multimodal_model_path="",
#         output_dir="",
#         classifier_scale=0,
#         devices=None,
#         is_strict=True,
#         all_save_num= 1024,
#         seed=42,
#         load_noise="",
#         data_dir="",
#         condition='celltype',
#         specific_type=None,
#     )
   
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# def load_gene2vec_txt_simple(path, dtype=np.float32):
#     with open(path, "r") as f:
#         num_genes, dim = map(int, f.readline().strip().split())
#         genes = []
#         embs = np.zeros((num_genes, dim), dtype=dtype)
#         for i, line in enumerate(f):
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             genes.append(parts[0])
#             embs[i] = np.asarray(parts[1:], dtype=dtype)
#     gene2idx = {g: i for i, g in enumerate(genes)}
#     return genes, embs, gene2idx



# def build_perturbation_embeddings(adata_pert,gene2vec_path):
#         genes, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
#         dim = embs.shape[1]
#         pert_series = adata_pert.obs["perturbation"].astype(str)
#         nperts_value = adata_pert.obs["nperts"].astype(int)

#         vectors = []
#         missing = []
#         for g, n in zip(pert_series, nperts_value):
#             if n == 1:
#                 if g in gene2idx:
#                     vectors.append(embs[gene2idx[g]])
#                 else:
#                     # 缺失时给 0；也可以改成小随机噪声
#                     vectors.append(np.zeros(dim, dtype=np.float32))
#                     missing.append(g)
#             elif n == 2:
#                 g1, g2 = g.split("_")
#                 vec = np.zeros(dim, dtype=np.float32)
#                 if g1 in gene2idx:
#                     vec += embs[gene2idx[g1]]
#                 else:
#                     missing.append(g1)
#                 if g2 in gene2idx:
#                     vec += embs[gene2idx[g2]]
#                 else:
#                     missing.append(g2)
#                 vectors.append(vec / 2)
#             else:
#                 print(f"[gene2vec] Warning: 发现 nperts > 2 的扰动 {g}")
                
#         pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
#         pert_emb_dim = dim
#         uniq_missing = sorted(set(missing))
#         if uniq_missing:
#             print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
#         print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
#         return pert_emb, pert_emb_dim
# # def build_perturbation_embeddings(adata_pert,gene2vec_path):
# #     genes, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
# #     dim = embs.shape[1]
# #     pert_series = adata_pert.obs["condition"].astype(str)

# #     vectors = []
# #     missing = []
# #     for g in pert_series:
# #         if g in gene2idx:
# #             vectors.append(embs[gene2idx[g]])
# #         else:
# #             # 缺失时给 0；也可以改成小随机噪声
# #             vectors.append(np.zeros(dim, dtype=np.float32))
# #             missing.append(g)

# #     pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
# #     pert_emb_dim = dim
# #     uniq_missing = sorted(set(missing))
# #     if uniq_missing:
# #         print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
# #     print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
# #     return pert_emb, pert_emb_dim




# if __name__ == "__main__":
#     print(th.cuda.current_device())
#     main()



































# #单diffusion基因扰动ood实验
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

# from scduo.scduo_perturbation.diffusion import dist_util, logger
# from scduo.scduo_perturbation.diffusion.multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict
# )
# from scduo.scduo_perturbation.diffusion.common import set_seed_logger_random, delete_pkl
# from scduo.scduo_perturbation.diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
# import scanpy as sc
# from distutils.spawn import spawn
# import random
# import blobfile as bf
# import numpy as np
# import torch as th
# import os
# import pickle
# import torch as th
# from torch.utils.data import DataLoader, Dataset

# import scanpy as sc
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# import yaml

# import muon as mu
# from muon import MuData
# from scipy.sparse import issparse
# from scduo.scduo_perturbation.vae.models.base.vae_model import EncoderModel
# import ot

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
#         sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
#         multimodal_model.specific_type = 0

#         logger.log(f"sampling samples for {model_path}")
#         model_name = model_path.split('/')[-1]

#         groups= 0
#         multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
#         sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
#         audio_save_path = os.path.join(args.output_dir)
#         img_save_path = os.path.join(args.output_dir)
#         #if dist.get_rank() == 0:
#         #单gpu
#         if dist_util.is_main_process():
#             os.makedirs(multimodal_save_path, exist_ok=True) 
#             os.makedirs(sr_save_path, exist_ok=True)
#             os.makedirs(audio_save_path, exist_ok=True)
#             os.makedirs(img_save_path, exist_ok=True)


#         # NK_dataset_path = "/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad"
#         adata_sc = sc.read(args.data_dir)
#         adata_ctrl = adata_sc[adata_sc.obs["cp_type"] == "control"].copy()
#         adata_pert = adata_sc[adata_sc.obs["cp_type"] == "stimulated"].copy()
#         adata = {
#             "ctrl": adata_ctrl,
#             "pert": adata_pert
#         }
#         adata_ood = sc.read("/home/wuboyang/scduo-new/dataset/gene_perturb_data/norman_ood.h5ad")

#         if args.class_cond:
#             from sklearn.preprocessing import LabelEncoder
#             labels = adata_ood.obs[args.condition].values
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             classes_all = label_encoder.transform(labels)

#         ae_path = "/home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v8.ckpt"
#         encoder_config = "/home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml"
#         gene2vec_path="/home/wuboyang/scduo-new/gene2vec/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
#         with open(encoder_config, 'r') as file:
#             yaml_content = file.read()
#         autoencoder_args = yaml.safe_load(yaml_content)

#         # Initialize encoder
#         encoder_model = EncoderModel(in_dim=adata_ctrl.shape[1],
#                                             n_cat=7,
#                                             conditioning_covariate=args.condition, 
#                                             encoder_type='learnt_autoencoder',
#                                             **autoencoder_args)
        
#         # Load weights 
#         encoder_model.load_state_dict(torch.load(ae_path, map_location=dist_util.dev())["state_dict"])
#         encoder_model.eval()

#         gene_emb, self_gene_emb_dim = build_perturbation_embeddings(adata_ood, gene2vec_path)


#         ctrl_X = adata_ctrl.X
#         pert_X = adata_ood.X
#         if issparse(ctrl_X):
#             ctrl_X = ctrl_X.toarray()
#         if issparse(pert_X):
#             pert_X = pert_X.toarray()
#         ctrl_X = torch.tensor(ctrl_X, dtype=encoder_model.dtype, device=encoder_model.device)
#         pert_X = torch.tensor(pert_X, dtype=encoder_model.dtype, device=encoder_model.device)
#         from scduo.scduo_perturbation.vae.data.utils import normalize_expression
#         ctrl_X_norm = normalize_expression(ctrl_X, ctrl_X.sum(), encoder_type='learnt_autoencoder')
#         pert_X_norm = normalize_expression(pert_X, pert_X.sum(), encoder_type='learnt_autoencoder')
#         z_ctrl,_,_ = encoder_model.encode({"X_norm":ctrl_X_norm})
#         z_pert,_,_ = encoder_model.encode({"X_norm":pert_X_norm})
#         z_ctrl = z_ctrl.detach().cpu().numpy()
#         z_pert = z_pert.detach().cpu().numpy()
#         M = ot.dist(z_pert, z_ctrl, metric='euclidean')
#         G = ot.emd(torch.ones(z_pert.shape[0]) / z_pert.shape[0],
#                     torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
#                     torch.tensor(M), numItermax=100000)
#         match_idx = torch.max(G, 1)[1].numpy()
#         ctrl_new = z_ctrl[match_idx]

#         ctrl = ctrl_new
#         pert = z_pert
        
#         npzfile = np.load('/'.join(ae_path.split('/')[:-2])+'/norm_factor9.npz')
#         ctrl_std10 = float(np.asarray(npzfile["ctrl_std"]))
#         pert_std10 = float(np.asarray(npzfile["pert_std"]))

#         audio_cond_all = np.expand_dims(ctrl / ctrl_std10, axis=1).astype(np.float32)
            
        
#         videos = []
#         audios = []
#         all_labels = []
#         ids = []
#         #target_idx = np.where(adata_ood.obs["perturbation"] == "SET_KLF1")[0]


#         try:
#             world_size = dist.get_world_size()
#         except ValueError:
#             world_size = 1

#         while groups * args.batch_size *  world_size< args.all_save_num: 
#             model_kwargs = {}
#             if args.class_cond:
#                 if args.specific_type is None:
#                     n = len(classes_all)
#                     #n = len(target_idx)
#                     if args.batch_size > n:
#                         raise ValueError(f"batch_size({args.batch_size}) 不能大于类别总数({n})")
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     #idx = np.random.choice(target_idx, size=args.batch_size, replace=False)
#                     classes = classes_all[idx]
#                     audio_cond = audio_cond_all[idx]
#                     audio_gene_cond = gene_emb[idx]
#                     ids.append(idx)
                    
#                    #classes = np.random.choice(classes_all, args.batch_size, replace=False)  # generated random cell type
#                 else:
#                     print(f'generating {int(args.specific_type)} cell')
#                     n = ctrl.shape[0]
#                     classes = th.ones(args.batch_size)*int(args.specific_type)   # generated certain cell type
#                     idx = np.random.choice(n, size=args.batch_size, replace=False)
#                     audio_cond = audio_cond_all[idx]
                    
#                 classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
#                 audio_cond = th.tensor(audio_cond, device=dist_util.dev(), dtype=th.float32)
#                 audio_gene_cond = th.tensor(audio_gene_cond, device=dist_util.dev(), dtype=th.float32)

#                 model_kwargs["label"] = classes
#                 model_kwargs["audio_cond"] = audio_cond
#                 model_kwargs["audio_gene_cond"] = audio_gene_cond
                

#             shape = {"video":(args.batch_size , *args.ctrl_dim), \
#                     "audio":(args.batch_size , *args.pert_dim)
#                 }
#             if args.sample_fn == 'dpm_solver':
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
#                     multimodal_diffusion.p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
#                 )

#                 sample = sample_fn(
#                     multimodal_model,
#                     shape = shape,
#                     clip_denoised=args.clip_denoised,
#                     model_kwargs=model_kwargs,
#                 )

#             audio = sample["audio"]              

#             all_audios = audio.detach().cpu().numpy()

#             if args.class_cond:
#                 all_labels.append(classes.cpu().numpy())
                
#             audios.append(all_audios)

#             groups += 1

#             #单gpu
#             if dist.is_initialized():
#                 dist.barrier()

#         audios = np.concatenate(audios)
#         all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
#         ids = np.concatenate(ids)       
#         output_path = os.path.join(img_save_path, f"sample_data.npz")

#         np.savez(output_path,pert=audios,label=all_labels,idx=ids)


#     logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         ref_path="",
#         batch_size=16,
#         sample_fn="dpm_solver",
#         multimodal_model_path="",
#         output_dir="",
#         classifier_scale=0,
#         devices=None,
#         is_strict=True,
#         all_save_num= 1024,
#         seed=42,
#         load_noise="",
#         data_dir="",
#         condition='celltype',
#         specific_type=None,
#     )
   
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# def load_gene2vec_txt_simple(path, dtype=np.float32):
#     with open(path, "r") as f:
#         num_genes, dim = map(int, f.readline().strip().split())
#         genes = []
#         embs = np.zeros((num_genes, dim), dtype=dtype)
#         for i, line in enumerate(f):
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             genes.append(parts[0])
#             embs[i] = np.asarray(parts[1:], dtype=dtype)
#     gene2idx = {g: i for i, g in enumerate(genes)}
#     return genes, embs, gene2idx



# def build_perturbation_embeddings(adata_pert,gene2vec_path):
#         genes, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
#         dim = embs.shape[1]
#         pert_series = adata_pert.obs["perturbation"].astype(str)
#         nperts_value = adata_pert.obs["nperts"].astype(int)

#         vectors = []
#         missing = []
#         for g, n in zip(pert_series, nperts_value):
#             if n == 1:
#                 if g in gene2idx:
#                     vectors.append(embs[gene2idx[g]])
#                 else:
#                     # 缺失时给 0；也可以改成小随机噪声
#                     vectors.append(np.zeros(dim, dtype=np.float32))
#                     missing.append(g)
#             elif n == 2:
#                 g1, g2 = g.split("_")
#                 vec = np.zeros(dim, dtype=np.float32)
#                 if g1 in gene2idx:
#                     vec += embs[gene2idx[g1]]
#                 else:
#                     missing.append(g1)
#                 if g2 in gene2idx:
#                     vec += embs[gene2idx[g2]]
#                 else:
#                     missing.append(g2)
#                 vectors.append(vec / 2)
#             else:
#                 print(f"[gene2vec] Warning: 发现 nperts > 2 的扰动 {g}")
                
#         pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
#         pert_emb_dim = dim
#         uniq_missing = sorted(set(missing))
#         if uniq_missing:
#             print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
#         print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
#         return pert_emb, pert_emb_dim
# # def build_perturbation_embeddings(adata_pert,gene2vec_path):
# #     genes, embs, gene2idx = load_gene2vec_txt_simple(gene2vec_path, dtype=np.float32)
# #     dim = embs.shape[1]
# #     pert_series = adata_pert.obs["condition"].astype(str)

# #     vectors = []
# #     missing = []
# #     for g in pert_series:
# #         if g in gene2idx:
# #             vectors.append(embs[gene2idx[g]])
# #         else:
# #             # 缺失时给 0；也可以改成小随机噪声
# #             vectors.append(np.zeros(dim, dtype=np.float32))
# #             missing.append(g)

# #     pert_emb = np.stack(vectors).astype(np.float32)  # shape: [N_pert, dim]
# #     pert_emb_dim = dim
# #     uniq_missing = sorted(set(missing))
# #     if uniq_missing:
# #         print(f"[gene2vec] 未在词表中找到的基因（{len(uniq_missing)}）示例1个：{uniq_missing[:1]}")
# #     print(f"[gene2vec] 已对齐 gene2vec：N={pert_emb.shape[0]}, dim={pert_emb_dim}")
# #     return pert_emb, pert_emb_dim




# if __name__ == "__main__":
#     print(th.cuda.current_device())
#     main()
