#!/bin/bash

python prepare_templates.py \
--post_process \
--template_labels_file ../templates/back_camera/hands/labels.csv \
--template_images_dir ../templates/back_camera/hands \
--save_dir ../templates/back_camera/binary_hands