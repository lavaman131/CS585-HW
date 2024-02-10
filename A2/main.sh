#!/bin/bash

python main.py \
--ground_truth_label 1 \
--template_labels_file "./templates/binary_images/labels.csv" \
--template_images_dir "./templates/binary_images" \
--save_dir "./experiments/demo" \
--roi_width 640 \
--roi_height 790 \
--gamma 0.375 \
--camera_id 1 \
--width 1920 \
--height 1080 \
--fps 30 \
--num_frames_to_save 5 \
--start_delay_seconds 3