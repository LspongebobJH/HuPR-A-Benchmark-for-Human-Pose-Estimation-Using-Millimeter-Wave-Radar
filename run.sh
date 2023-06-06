#!/bin/bash

# for ((i=11; i<=14; i++)); do
#     python preprocessing/process_iwr1843.py --index $i &
# done

# python preprocessing/process_iwr1843.py --index_single 46 --frame 347 &
# python preprocessing/process_iwr1843.py --index 2 &
# python preprocessing/process_iwr1843.py --index 3 &
# python preprocessing/process_iwr1843.py --index 4 &
# python preprocessing/process_iwr1843.py --index 7 &


# python preprocessing/process_iwr1843.py

# python main.py --config mscsa_prgcn.yaml --eval

# python gen_gt.py
HYDRA_FULL_ERROR=1

python main.py RUN.use_ray=True SETUP.numWorkers=4 TRAINING.epochs=30 RUN.num_ray_workers=4

