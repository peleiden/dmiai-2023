#!/bin/sh
#BSUB -q gpua100
#BSUB -J 05
#BSUB -R select[gpu40gb]
#BSUB -gpu num=1:mode=exclusive_process
#BSUB -R rusage[mem=15GB]
#BSUB -n 4
#BSUB -W 24:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183912@dtu.dk
#BSUB -N

source ~/.hpcrc

python3 tumor_segmentation/train.py /work3/s183912/tooomaaah -c tumor_segmentation/jobs/05.ini
