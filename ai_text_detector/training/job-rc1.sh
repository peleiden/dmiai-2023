#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1::mode=exclusive_process"
##BSUB -R "select[gpu80gb]"

#BSUB -n 4
#BSUB -R "rusage[mem=50GB]"
#BSUB -R "span[hosts=1]"

#BSUB -W 24:00

#BSUB -J "rc1"
#BSUB -N
#BSUB -u s183911@student.dtu.dk
#BSUB -oo /zhome/ac/c/137651/joblogs/stdout_%J.out
#BSUB -eo /zhome/ac/c/137651/joblogs/stderr_%J.out

echo "Starting job on GPU $CUDA_VISIBLE_DEVICES ..."

TPATH=/work3/s183911/dmiai
source /zhome/ac/c/137651/dmiai-setup.sh

python ai_text_detector/training/hf_loop.py\
    $TPATH/rc1\
    -c ai_text_detector/training/moar-epochs.ini\
    --weight-decay 0.0\
    --scheduler linear\
    --warmup-prop 0.25\
    --epochs 4\
    --n-ensemble 3



echo "Finished job !"
