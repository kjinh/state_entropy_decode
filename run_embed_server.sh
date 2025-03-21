#!/bin/bash
#SBATCH --job-name=embed_server
#SBATCH --gres=gpu:2             
#SBATCH --partition=suma_rtx4090
#SBATCH --mem=64G
#SBATCH --time=40:00:00

# set
cd ~/state_entropy_decode
source ~/.bashrc
conda activate sedecode

export CONFIG="recipes/Llama-3.2-1B-Instruct/beam_search.yaml"


# 서버 실행
python prm_embed_server.py > logs/prm_embed.log 2>&1
