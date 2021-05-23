#!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J maddpg
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -R "span[hosts=1]"
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 module load python3/3.6.2
 echo "Traning..."
 python3 main.py -c "maddpg_ckpt"
