#!/bin/bash
#SBATCH --job-name=my_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scLDM
export PYTHONNOUSERSITE=1


python train_vae.py \
  dataset=covid \
  logger.project=my_vae \
  trainer.max_epochs=500 \
  dataset.encoder_type=learnt_autoencoder \
  encoder.learning_rate=0.001 \
  encoder.weight_decay=0.0001 \
  encoder.kl_weight=5e-4 \
  encoder.noise_rate=0.1 \
 
