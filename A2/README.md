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

### Classical Computer Vision Algorithms

I use binary image analysis followed by max contour detection for the segmentation of the hand. I also use template matching (with templates augmented via rotations to capture possible orientations of the hand) with the maximum normalized correlation coefficient for classifying the hand movement as the digit 1, 2, 3, 4, or 5.

#### Binary Image Conversion Using Skin Color

!TODO add more details about the algorithms used for the following including mathematical details and algorithm

#### Countour Detection for Hand Segmentation

!TODO add more details about the algorithms used for the following including mathematical details and algorithm

#### Template Matching for Sign Language Digit Classification

!TODO add more details about the algorithms used for the following including mathematical details and algorithm

## 🔬 Experiments

I conducted the following experiments to evaluate the performance of the hand gesture recognition system and logged the results in the `./experiments` directory with [hydra](https://hydra.cc/docs/intro/) configuration files. See more details about this in the [usage section](#-usage) of the README.

## 📈 Results

I ran 20 trials with each finger to evaluate the performance of the hand gesture recognition system. For instance, I used the following command for the sign language digit 5 with 20 frames and a 5-second delay to start the camera and allow the user to prepare their hand for the gesture recognition system to capture the frames and process them for classification:

```bash
python main.py \
hydra.job.name=class_5 \
processing.ground_truth_label=5 \
camera.num_frames=20 \
camera.start_delay_seconds=5
```

The following are the results of the experiments which I logged in the `./experiments` directory. Then I used the `./tools/evaluate.py` script to evaluate the performance of the hand gesture recognition system and logged the results in the `./reports` directory.


### Confusion Matrix

<img src="./reports/confusion_matrix.png" width="50%">

Along with the program, submit the following information about your graphics program:

An overall description
How the graphics respond to different hand shapes and/or gestures
Interesting and fun aspects of the graphics display

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

To run the live demo follow the instructions below. The demo will start the camera and display the live feed. The program will then wait for a few seconds (`start_delay_seconds`) to allow the camera to adjust to the lighting conditions and user to prepare the motion. **Note, that only content inside of the green bounding box will be processed, so that is where the user should put their hand.** After the delay, the program will start capturing frames (`num_frames_to_save`) and processing them. The program will display the processed frames and the classification results in real-time. The program will also save the binary image and processed frames to the specified directory (`save_dir`).

### Slight Technical Limitations

The program works best when the user tries to shape their hand to mimic the template images (`./templates/binary_images`) for the sign-language digit (1-5) that they are trying to automatically classify. The hand should be still and the background should be relatively static with not too much overexposure or underexposure in the camera. The program will not work well if the hand is in motion or if the background is not relatively static.

### Visualization of GUI

The program will display the following GUI:

![GUI](./reports/gui.png)

### Basic Usage

```bash
python main.py
```

### Advanced Usage

Refer to `predict` function in `main.py` for more details about parameters. You can specify the following parameters in the `conf/config.yaml` hydra configuration file or as command line arguments, e.g.

```bash
python main.py \
hydra.job.name=rotation_template_matching \
processing.ground_truth_label=1
```

## 🗣️ Discussion

## 🏆 Conclusions

## 🎬 Credits and Bibliography

[Gamma Correction](https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

[Count Approximation](https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/)

## 👥 Collaborators

* None