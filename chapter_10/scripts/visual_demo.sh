#!/usr/bin/env bash

tracker=${1:-'ostrack'}
tracker_param=${2:-'vitb_256_mae_ce_32x4_ep300'}
videofile=${3:-'comet_new.mp4'}

CUDA_VISIBLE_DEVICES=0 \
nohup \
python tracking/video_demo.py --tracker_name $tracker --tracker_param $tracker_param \
                              --videofile $videofile --save_results \
> terminal_logs/demo_$tracker'_'$tracker_param'_'$videofile.log 2>&1 &

echo log save to terminal_logs/demo_$tracker'_'$tracker_param'_'$videofile.log
#python tracking/test.py --tracker_name $script --tracker_param $config --dataset $dataset \
#                        --threads $((threads_per_gpu*numgpu)) --num_gpus $numgpu \
#                        --search_param $hyperparam --debug 0 \