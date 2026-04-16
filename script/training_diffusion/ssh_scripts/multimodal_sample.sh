export CUDA_VISIBLE_DEVICES=1
MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True
--ctrl_dim 1,100 --pert_dim 1,100 --learn_sigma False --num_channels 128 
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True  --class_cond True --condition cell_type --num_class 1 "
#--specific_type 6

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 200  --devices 1
--batch_size 2000  --is_strict True --sample_fn ddim"

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs25/train_outputs/model200000.pt"
OUT_DIR="/home/wuboyang/scduo-new/script/training_diffusion/outputs/outputs25/sample_outputs"
DATA_DIR="/home/wuboyang/scduo-new/dataset/gene_perturb_data/adamson.h5ad"

NUM_GPUS=1
python3 py_scripts/multimodal_sample.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH} --data_dir ${DATA_DIR}
