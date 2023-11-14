#!/bin/sh
#BSUB -q gpuh100
#BSUB -gpu "num=1::mode=exclusive_process"
##BSUB -R "select[gpu32gb]"

#BSUB -n 4
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"

#BSUB -W 8:00

#BSUB -J "tumor"
#BSUB -N
#BSUB -u s183911@student.dtu.dk
#BSUB -oo /zhome/ac/c/137651/joblogs/stdout_%J.out
#BSUB -eo /zhome/ac/c/137651/joblogs/stderr_%J.out

echo "Starting job on GPU $CUDA_VISIBLE_DEVICES ..."

TPATH=/work3/s183911/toomah
TPATH=/zhome/ac/c/137651/dmiai/local-data/toomah
export HUGGINGFACE_HUB_CACHE=/zhome/ac/c/137651/.shitty_hf_cache
source /zhome/ac/c/137651/dmiai-setup.sh
export PYTHONPATH=$PYTHONPATH:.

python tumor_segmentation/train.py\
    $TPATH\
    -c tumor_segmentation/jobs/standard-nova.ini



echo "Finished job !"
