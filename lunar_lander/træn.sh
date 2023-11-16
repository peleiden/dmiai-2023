#!/bin/sh
#BSUB -q epyc
#BSUB -J train
#BSUB -R rusage[mem=2GB]
#BSUB -n 64
#BSUB -W 24:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183912@dtu.dk
#BSUB -N

#source ~/.hpcrc
source ~/dmiai-setup.sh
pip3 install gymnasium[box2d]
python3 multitrainer.py
# python3 eval.py

