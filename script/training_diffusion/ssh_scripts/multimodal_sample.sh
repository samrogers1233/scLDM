# ---------- edit these ----------
DATA_DIR="path/to/dataset.h5ad"
OUTPUT_DIR="path/to/sample_outputs"
MODEL_PATH="path/to/model200000.pt"
ENCODER_CONFIG="path/to/encoder/default.yaml"
AE_PATH="path/to/autoencoder.ckpt"
GENE2VEC_PATH="path/to/gene2vec.txt"
NORM_FACTOR_NAME="norm_factor19.npz"

# Mode: iid | ood. When ood, also set OOD_DATA_DIR below.
MODE=iid
OOD_DATA_DIR=""

# GPU: pick one physical GPU and let the script see it as device 0
export CUDA_VISIBLE_DEVICES=0
DEVICES=0

# ---------- model / sampling ----------
MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8 \
--cross_attention_shift True \
--ctrl_dim 1,100 --pert_dim 1,100 --learn_sigma False --num_channels 128 \
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False \
--use_scale_shift_norm True --class_cond True --condition cell_type --num_class 1"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear \
--devices ${DEVICES} --batch_size 2000 --is_strict True --sample_fn ddim"

# ---------- run ----------
cd "$(dirname "$0")/.."
python3 py_scripts/gene_sample.py \
  $MODEL_FLAGS $DIFFUSION_FLAGS \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --multimodal_model_path ${MODEL_PATH} \
  --mode ${MODE} \
  --ood_data_dir "${OOD_DATA_DIR}" \
  --encoder_config ${ENCODER_CONFIG} \
  --ae_path ${AE_PATH} \
  --gene2vec_path ${GENE2VEC_PATH} \
  --norm_factor_name ${NORM_FACTOR_NAME}
