# ---------- edit these ----------
DATA_DIR="path/to/dataset.h5ad"
OUTPUT_DIR="path/to/train_outputs"
ENCODER_CONFIG="path/to/encoder/default.yaml"
AE_PATH="path/to/autoencoder.ckpt"

# GPU: pick one physical GPU and let the script see it as device 0
export CUDA_VISIBLE_DEVICES=0
DEVICES=0

# ---------- model / training / diffusion ----------
MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8 \
--cross_attention_shift True --dropout 0.0 \
--ctrl_dim 1,100 --pert_dim 1,100 --learn_sigma False \
--num_channels 128 --num_head_channels -1 --num_res_blocks 1 \
--resblock_updown True --use_fp16 False --use_scale_shift_norm True \
--num_workers 4 --condition cell_type --num_class 1 \
--weight_decay 0.0001 --use_gene_cond True \
--encoder_config ${ENCODER_CONFIG} --ae_path ${AE_PATH}"

TRAIN_FLAGS="--lr 0.0001 --batch_size 64 --devices ${DEVICES} \
--log_interval 100 --save_interval 200000 --use_db False --lr_anneal_steps=200000"

DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --sample_fn dpm_solver++"

# ---------- run (resolve paths relative to this script so it works from anywhere) ----------
cd "$(dirname "$0")/.."
python3 py_scripts/multimodal_train.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS
