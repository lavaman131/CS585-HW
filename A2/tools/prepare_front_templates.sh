#!/bin/bash

python prepare_templates.py \
--template_labels_file ../templates/front_face_camera/hands/labels.csv \
--template_images_dir ../templates/front_face_camera/hands \
--save_dir ../templates/front_face_camera/binary_hands