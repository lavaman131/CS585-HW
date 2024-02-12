#!/bin/bash

camera_id=$1
ground_truth_label=$2

# if ground_truth_label is not provided, set it to -1
if [ -z "$ground_truth_label" ]
then
  ground_truth_label=-1
fi

python main.py \
--template_labels_file "./templates/binary_images/labels.csv" \
--template_images_dir "./templates/binary_images" \
--save_dir "./experiments/demo" \
--ground_truth_label $ground_truth_label \
--roi_width 640 \
--roi_height 790 \
--gamma 0.375 \
--camera_id $camera_id \
--width 1920 \
--height 1080 \
--fps 30 \
--num_frames_to_save 5 \
--start_delay_seconds 3