#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --gres=gpu:1            
#SBATCH --partition=suma_rtx4090
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=logs/run_exp.out
#SBATCH --error=logs/run_exp.err



cd ~/state_entropy_decode

export CONFIG="~/state_entropy_decode/recipes/Llama-3.2-1B-Instruct/beam_search.yaml"

source ~/.bashrc
conda activate sedecode

for server in "run_embed_server.sh" "run_reward_server.sh" "run_gen_server.sh"; do
    sbatch ${server}
done

echo "Waiting for servers to start..."
sleep 350

echo "Starting experiment..."
python scripts/test_time_compute.py $CONFIG


echo "Experiment completed!"
exit
