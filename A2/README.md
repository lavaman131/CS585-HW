# ✋ Hand Gesture Recognition Project

## 🎯 Learning Objectives
- **Video Processing**: Read and display video frames from a webcam.
- **Template Matching**: Learn about tracking by template matching.
- **Image Analysis**: Dive into analyzing properties of objects in an image, like object centroid, axis of least inertia, and shape (circularity).
- **Graphical Applications**: Create engaging and interactive graphical applications.

## 📋 Requirements
Your mission is to design and implement algorithms capable of recognizing hand shapes (e.g., making a fist, thumbs up/down, pointing) or gestures (e.g., waving, swinging, drawing in the air) and create a graphical display that responds to these recognitions. Utilize at least two of the following computer vision techniques discussed in class for binary object shape analysis:
- Bounding boxes identification using horizontal and vertical projections.
- Analysis of size, position, and orientation of the object of interest.
- Object circularity.
- Template matching for different hand shapes.
- Background and frame-to-frame differencing for motion detection.
- Motion energy templates and skin-color detection.
- Tracking the position and orientation of moving objects.

Feel free to use OpenCV library functions, but ensure you understand their workings in detail.

## 🎲 Algorithm Requirements
- Detect at least four distinct hand shapes or gestures.
- Include a detailed explanation of the mathematical formulations and algorithms used.

## 📄 Report Submission
Submit a detailed report including:
- A confusion matrix illustrating the classification accuracy of your system.
- An engaging description of how the graphical display responds to different gestures.
- Highlights of interesting and fun aspects of your graphical display.

## 🎥 Demo Submission
Along with your code and report, submit a real-time demo video showcasing your project in action. If you encounter issues uploading the video directly, provide a link to Google Drive or an unlisted YouTube video (ensure proper viewing permissions).

## 📚 Submission Guidelines
- **Code**: Submit to Gradescope under "A2".
- **Report**: Individual effort required, even for team projects.
- **Video Demo**: Teams may submit one shared video.
- Ensure all team members are mentioned in submissions.

🌟 Let your creativity shine through your graphical display and have fun with this project! 🌟

## Collecting Data

```python
python collect_data.py --label "demo" --save_dir "data" --width 224 --height 224
```