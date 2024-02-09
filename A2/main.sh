#!/bin/bash

python main.py \
--save_dir "./experiments/demo" \
--labels_file "./templates/binary_images/labels.csv" \
--template_images_dir "./templates/binary_images" \
--rectangle_width 640 \
--rectangle_height 790 \
--gamma 0.375 \
--camera_id 1 \
--width 640 \
--height 480 \
--fps 30 \
--num_frames_to_save 5 \
--start_delay_seconds 3