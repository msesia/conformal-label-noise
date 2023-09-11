#!/bin/bash

module purge
eval "$(conda shell.bash hook)"
conda activate default

export OPENBLAS_NUM_THREADS=1

python3 exp_4.py $1 $2 $3 $4 $5 $6 $7 $8 $9 "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}"
