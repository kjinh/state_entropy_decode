#!/bin/bash
#SBATCH --job-name=gen_server
#SBATCH --gres=gpu:1            
#SBATCH --partition=suma_rtx4090
#SBATCH --mem=64G
#SBATCH --time=40:00:00

# set
cd /home/mjmps0725/state_entropy_decode
source ~/.bashrc
conda activate sedecode

export CONFIG="/home/mjmps0725/state_entropy_decode/recipes/Llama-3.2-1B-Instruct/beam_search.yaml"


# 서버 실행
python gen_server.py > logs/gen.log 2>&1
