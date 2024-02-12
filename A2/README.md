# 🖖 Hand Gesture Recognition Project

## 📚 Problem Definition

The problem is to recognize sign-language hand gestures from a video stream. This is useful because it can be used to create human computer interfaces that are more accessible to people with hearing disabilities. My analysis assumes that the background is relatively static and that the hand is the only moving object in the video stream.

Some difficulties that I anticipate are:

- The hand can be in different orientations and positions in the video stream.
- The hand can be in different lighting conditions.
- The hand can be occluded by other objects in the video stream.
- The hand can be in motion.

The gestures are defined as follows:

- **One**: The thumb is extended and the other fingers are closed.
- **Two**: The thumb and the index finger are extended and the other fingers are closed.
- **Three**: The thumb, index finger, and middle finger are extended and the other fingers are closed.
- **Four**: The thumb, index finger, middle finger, and ring finger are extended and the little finger is closed.
- **Five**: All fingers are extended.

## 🛠️ Method and Implementation

### Project Structure

The project is structured as follows:

```
.
├── README.md
├── a2
│   ├── __init__.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   └── segmentation.py
│   └── data
│       ├── __init__.py
│       ├── preprocessing.py
│       └── utils.py
├── experiments
│   ├── black_background
│   │   ├── ...
│   ├── demo
│   │   ├── ...
│   └── white_background
│       ├── ...
├── main.py
├── main.sh
├── pyproject.toml
├── templates
│   ├── binary_images
│   │   ├── 1.png
│   │   ├── ...
│   │   └── labels.csv
│   └── original
│       ├── 1.png
│       ├── ...
│       └── labels.csv
└── tools
    ├── prepare_templates.py
    └── prepare_templates.sh
```

I use binary image analysis followed by max contour detection for the segmentation of the hand. I also use template matching (with templates augmented via rotations to capture possible orientations of the hand) with the maximum normalized correlation coefficient for classifying the hand movement as the digit 1, 2, 3, 4, or 5.

## 🔬 Experiments

## 📈 Results

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

## 🚀 Usage

To run the demo, execute the following command:

```bash
./main.sh
```

## 🗣️ Discussion

## 🏆 Conclusions

## 🎬 Credits and Bibliography

[Gamma Correction](https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

[Count Approximation](https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/)