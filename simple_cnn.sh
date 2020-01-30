#!/bin/bash
#SBATCH --account=def-lemc2220
#SBATCH --gres=gpu:1         # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=30000M         # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:10     # DD-HH:MM:SS

cd $SLURM_TMPDIR

module load python/3.6 cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision

mkdir $SLURM_TMPDIR/tin
tar xf /home/lemc2220/projects/def-pmjodoin/lemc2220/data/tinyimagenet/tinyimagenet.tar -C $SLURM_TMPDIR/tin  # Transfer all data

python ~/source/cours-ai-pmj/main.py $SLURM_TMPDIR/tin
