#!/bin/bash

python save_segmented_frames.py \
--save_dir "../data/demo" \
--rectangle_width 640 \
--rectangle_height 790 \
--gamma 0.375 \
--camera_id 1 \
--width 640 \
--height 480 \
--fps 30 \
--num_frames_to_save 5 \
--start_delay_seconds 3