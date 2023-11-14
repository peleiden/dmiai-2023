#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1::mode=exclusive_process"
##BSUB -R "select[gpu80gb]"

#BSUB -n 4
#BSUB -R "rusage[mem=50GB]"
#BSUB -R "span[hosts=1]"

#BSUB -W 5:00

#BSUB -J "moar-epochs"
#BSUB -N
#BSUB -u s183911@student.dtu.dk
#BSUB -oo /zhome/ac/c/137651/joblogs/stdout_%J.out
#BSUB -eo /zhome/ac/c/137651/joblogs/stderr_%J.out

echo "Starting job on GPU $CUDA_VISIBLE_DEVICES ..."

TPATH=/work3/s183911/dmiai
source /zhome/ac/c/137651/dmiai-setup.sh

python ai_text_detector/training/hf_loop.py\
    $TPATH/constant-lr\
    -c ai_text_detector/training/moar-epochs.ini\
    --weight-decay 0.0\
    --scheduler constant



echo "Finished job !"
