#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1::mode=exclusive_process"
##BSUB -R "select[gpu80gb]"

#BSUB -n 4
#BSUB -R "rusage[mem=50GB]"
#BSUB -R "span[hosts=1]"

#BSUB -W 5:00

#BSUB -J "tumor"
#BSUB -N
#BSUB -u s183911@student.dtu.dk
#BSUB -oo /zhome/ac/c/137651/joblogs/stdout_%J.out
#BSUB -eo /zhome/ac/c/137651/joblogs/stderr_%J.out

echo "Starting job on GPU $CUDA_VISIBLE_DEVICES ..."

TPATH=/work3/s183911/toomah
source /zhome/ac/c/137651/dmiai-setup.sh
export PYTHONPATH=$PYTHONPATH:.

python tumor_segmentation/train.py\
    $TPATH\
    --batches 20\
    --batch-size 8



echo "Finished job !"
