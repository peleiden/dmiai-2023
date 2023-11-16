#!/bin/sh
#BSUB -q gpuh100
#BSUB -J text
#BSUB -R select[gpu80gb]
#BSUB -gpu num=1:mode=exclusive_process
#BSUB -R rusage[mem=15GB]
#BSUB -n 4
#BSUB -W 24:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183912@dtu.dk
#BSUB -N

source ~/.hpcrc

python3 text_hpc.py .
