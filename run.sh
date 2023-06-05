#!/bin/bash

# for ((i=11; i<=14; i++)); do
#     python preprocessing/process_iwr1843.py --index $i &
# done

# python preprocessing/process_iwr1843.py --index_single 5 --frame 395 &
# python preprocessing/process_iwr1843.py --index 2 &
# python preprocessing/process_iwr1843.py --index 3 &
# python preprocessing/process_iwr1843.py --index 4 &
# python preprocessing/process_iwr1843.py --index 7 &


# python preprocessing/process_iwr1843.py

# python main.py --config mscsa_prgcn.yaml --eval

# python gen_gt.py
python main.py RUN.debug=True RUN.use_ray=False SETUP.numWorkers=4

