# object detection and tracking using deepsort & yolov8

# Tracking Object Behaviour

## About
This project focuses on tracking object behavior by leveraging a single camera sensor placed on road infrastructure. 
The notebook and code perform various tasks like object detection and trajectory tracking using YOLOv8 and DeepSORT, collision risk analysis, and more.

### Implemented Features
- Object Detection and Trajectory Tracking using YOLOv8 (Segmentation with DeepSORT)
- Object Detection Validation
- Future Trajectory Prediction
- Collision Risk Analysis
- Real-time Collision Risk Analysis
- Real-time Collision Risk Analysis based on future vehicle movement (6 UseCases)
- Collision Risk Validation
- Traffic Management: Count of Vehicles
- Traffic Management: Congestion

## Introduction
**INDIVIDUAL RESERACH THESIS: TRACKING OBJECT BEHAVIOUR BY LEVERAGING A SINGLE CAMERA SENSOR PLACED ON THE ROAD INFRASTRUCTURE**

NAME: CHANDNI SAHA

STUDENT ID: s387796

MSc in COMPUTATIONAL AND SOFTWARE TECHNIQUES IN ENGINEERING (COMPUTATIONAL INTELLIGENCE IN DATA ANALYTICS)

# Object Detection and Tracking with DeepSORT and YOLOv8

This repository provides a comprehensive guide to Object Detection and Tracking using DeepSORT and YOLOv8. We also include modules for future trajectory prediction, collision risk analysis, and traffic management.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Running the Modules](#running-the-modules)
5. [Visualization](#visualization)

---

## Installation

### Clone the Repository

\```bash
!git clone https://github.com/Chandni12300/object_detection_and_tracking_deepsort_yolov8.git
\```

### Install Ultralytics Library

\```bash
%pip install ultralytics
import ultralytics
ultralytics.checks()
\```

### Install Other Dependencies

\```bash
%cd /content/object_detection_and_tracking_deepsort_yolov8
!pip install -e '.[dev]'
\```

---

## Dataset Preparation

### Download COCO Dataset

\```bash
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="VJDeMwjnn3DSw8bIdkPJ")
project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
dataset = project.version(2).download("yolov5")
\```

### Conversion from YOLOv5 to YOLOv8

Use the `data.yaml` file for conversion. The folder name should be `vehicles-2`.

---

## Training

\```bash
!yolo train model=yolov8l.pt data=/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/vehicles-2/data.yaml epochs=50 imgsz=640
\```

---

## Running the Modules

### Object Detection and Tracking

\```bash
!python predict.py model=yolov8l.pt source='test1.mp4'
\```

### Future Trajectory Prediction

\```bash
!pip install filterpy
!python futuretrajectoryprediction.py model=yolov8l.pt source='test1.mp4'
\```

### Collision Risk Analysis

\```bash
!python riskanalysis.py model=yolov8l.pt source='test1.mp4'
\```

... (other modules)

---

## Visualization

### Displaying Graphs

\```python
from IPython.display import display, HTML
# ... (code to display graphs)
\```

### Displaying Images

\```python
from IPython.display import Image
# ... (code to display images)
\```


