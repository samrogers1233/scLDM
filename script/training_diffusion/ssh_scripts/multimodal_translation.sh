MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True
--ctrl_dim 1,100 --pert_dim 1,100 --learn_sigma False --num_channels 128
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --class_cond True"

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--devices 0 --classifier_scale 3.0
--batch_size 16   --is_strict True --sample_fn ddpm"

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="/home/wuboyang/scduo-main/script/training_diffusion/outputs/checkpoints/my_dfbackbone/output6/model800000.pt"
AE_PATH="/home/wuboyang/scduo-main/script/training_vae/outputs/checkpoints/my_vae/checkpoints/last-v6.ckpt"
OUT_DIR="/home/wuboyang/scduo-main/script/training_diffusion/outputs/checkpoints/my_dfbackbone/output6/translation_output" 

# translation config
DATA_DIR="/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_oodNK.h5ad"  # the source data
GEN_MODE="pert"     # the target modality
CONDITION="cell_type"  # the condition type to guide the generation
encoder_config="/home/wuboyang/scduo-main/script/training_vae/configs/encoder/default.yaml"    # the cfgen autoencoder config
gen_times="1"    # how many time you want to translate the whole data. usually translate more than once to remove noise.

NUM_GPUS=1
# mpiexec -n $NUM_GPUS 
CUDA_VISIBLE_DEVICES=1 python3 py_scripts/multimodal_translation.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH} --data_dir ${DATA_DIR} --gen_mode ${GEN_MODE} \
--condition ${CONDITION} --encoder_config ${encoder_config} --ae_path ${AE_PATH} --gen_times ${gen_times} --num_class 6
