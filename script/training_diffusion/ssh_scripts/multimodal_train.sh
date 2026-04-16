# MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
# --cross_attention_shift True --dropout 0.0 
# --ctrl_dim 1,100 --pert_dim 1,100 --learn_sigma False --num_channels 128
# --num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
# --use_scale_shift_norm True --num_workers 4 --condition cell_type
# --encoder_config /home/wuboyang/scduo-main/script/training_vae/configs/encoder/default.yaml
# --num_class 6 --weight_decay 0.0001
# --ae_path /home/wuboyang/scduo-main/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v1.ckpt
# "

# # Modify --devices to your own GPU ID
# TRAIN_FLAGS="--lr 0.0001 --batch_size 64
# --devices 0,1 --log_interval 100 --save_interval 200000 --use_db False --lr_anneal_steps=800000" 
# DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --sample_fn dpm_solver++" 

# # Modify the following pathes to your own paths
# DATA_DIR="/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_oodNK.h5ad"
# OUTPUT_DIR="/home/wuboyang/scduo-main/script/training_diffusion/outputs/checkpoints/my_dfbackbone/output1"
# NUM_GPUS=2
# WORLD_SIZE=1
# NCCL_P2P_DISABLE=1

# mpiexec -n $NUM_GPUS  python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 


MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
--cross_attention_shift True --dropout 0.0 
--ctrl_dim 1,100 --pert_dim 1,100 --learn_sigma False --num_channels 128
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --num_workers 4 --condition cell_type
--encoder_config /home/wuboyang/scduo-new/script/training_vae/configs/encoder/default.yaml
--num_class 1 --weight_decay 0.0001
--ae_path /home/wuboyang/scduo-new/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v18.ckpt
"

# Modify --devices to your own GPU ID
TRAIN_FLAGS="--lr 0.0001 --batch_size 64
--devices 1 --log_interval 100 --save_interval 200000 --use_db False --lr_anneal_steps=200000" 
DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --sample_fn dpm_solver++" 

# Modify the following pathes to your own paths
DATA_DIR="/home/wuboyang/scduo-new/dataset/gene_perturb_data/adamson.h5ad"
OUTPUT_DIR="/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs25/train_outputs"
NUM_GPUS=2
WORLD_SIZE=1
NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
cd /home/wuboyang/scduo-new/script/training_diffusion
python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 
