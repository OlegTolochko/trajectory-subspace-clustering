#!/usr/bin/env bash
#
#SBATCH --job-name=trajectory_embedding
#SBATCH --output=res_trajectory_embedding.txt
#SBATCH --ntasks=1
#SBATCH --time=5-23:00:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

env

# venv
source /home/wiss/xian/venvs/subspace_clustering_env/bin/activate
export BLAS=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3
export LAPACK=/usr/lib/x86_64-linux-gnu/lapack/liblapack.a
# export PATH="$PATH:/home/wiss/xian/Python_code/oleg/trajectory_embedding_learning/trajectory-subspace-clustering"
# echo $PATH
# pip install -U pip setuptools wheel
# train
python3 ./playground/inference_save_feature_embeddings.py  >> output_inference.txt
