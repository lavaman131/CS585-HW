# 👣 Single Object Tracking, Multi-Object Tracking, and Data Association

## 📚 Problem Definition

### Single Object Tracking

In the realm of computer vision and video processing, tracking the movement of objects across a sequence of frames is a fundamental challenge. One specific aspect of this challenge is Single Object Tracking (SOT), where the goal is to follow the trajectory of a single entity, such as a vehicle, across multiple frames.

The core task is to process detected 2D locations (x, y coordinates) of a vehicle as it moves across various frames of a video. These detections are inherently noisy due to various factors like camera movement, changes in lighting, and occlusions. The objective is to implement a filtering technique that can generate a smooth and continuous 2D track of the vehicle, mitigating the noise and inaccuracies in the observed locations. The choice of an alpha-beta filter or a Kalman filter allows for a balance between simplicity and effectiveness, leveraging predictions and measurements to estimate the vehicle's current and future states with a higher degree of accuracy.

### Multi-Object Tracking and Data Association

Expanding the challenge to Multi-Object Tracking (MOT), this project addresses the complexity of tracking multiple objects simultaneously. In real-world scenarios, a video may contain several objects of interest, each requiring tracking and identification over time. The task involves processing bounding boxes for multiple detected objects across video frames and efficiently tracking them by assigning a unique ID to each object. This ID must persist as long as the object is present and detected in the video.

The critical challenge in MOT is the Data Association problem, which involves correctly associating the detected bounding boxes with the corresponding tracks (or IDs) over time, especially in scenarios where objects may intersect, occlude each other, or leave and re-enter the frame. This requires sophisticated algorithms that can manage uncertainties and ambiguities in object detection and identification, ensuring accurate and consistent tracking of multiple objects throughout the video sequence.

This project aims to develop and implement solutions for both SOT and MOT, leveraging filtering techniques and data association strategies to achieve robust and reliable object tracking in video data.

## 🛠️ Method and Implementation

### Project Structure

The project is structured as follows:

```
.
├── README.md
├── a3
│   ├── __init__.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── object_detection.py
│   │   └── object_tracking.py
│   └── utils
│       ├── __init__.py
│       ├── draw
│       │   ├── object_detection.py
│       │   └── object_tracking.py
│       └── io
│           ├── __init__.py
│           ├── object_detection.py
│           └── object_tracking.py
├── data
│   ├── cropped
│   │   ├── ...
│   ├── original
│   │   ├── ...
│   └── submission
│       ├── ...
├── main.py
├── poetry.lock
├── pyproject.toml
└── tools
    └── prepare_data.py
```

## 🎮 Demo

### 📦 Setup

The project is implemented in Python 3.10. The dependencies are managed using Poetry. To install the dependencies, run the following commands:

```bash
# create a virtual environment
python -m venv .venv
# activate the virtual environment
source .venv/bin/activate
# install the dependencies
poetry install
```

## 📈 Results

### Single Object Tracking

![](./assets/sot.gif)

### Multi-Object Tracking

![](./assets/mot.gif)

## 👥 Collaborators

* None