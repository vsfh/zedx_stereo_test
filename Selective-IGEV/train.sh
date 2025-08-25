#!/bin/bash
module load miniforge3/24.11
source activate ultra
python train_stereo.py --restore_ckpt /data/home/sczd617/run/code/zedx_stereo_test/Selective-IGEV/sceneflow_igev.pth