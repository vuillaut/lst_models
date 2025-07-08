#!/bin/sh

apptainer run \
    --nv \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    --env "NUMBA_CACHE_DIR=/tmp/NUMBA" \
    --mount type=bind,source=/lapp_data/,destination=/lapp_data/ \
        --mount type=bind,source=/mustfs/LAPP-DATA/,destination=/mustfs/LAPP-DATA/ \
    /lapp_data/cta/vuillaume/lst_models/lst_models_main.sif /opt/conda/envs/torch/bin/python regressor_sweep.py
