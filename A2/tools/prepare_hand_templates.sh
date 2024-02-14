#!/bin/bash

python prepare_templates.py \
--template_labels_file ../templates/hands/labels.csv \
--template_images_dir ../templates/hands \
--post_process \
--save_dir ../templates/binary_hands