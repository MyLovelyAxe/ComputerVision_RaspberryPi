# Computer Vision tasks on Raspberry Pi 5

This repository implement embedded computer vision solutions on Rasberry Pi OS with access of [camera module v3](https://www.raspberrypi.com/products/camera-module-3/).


<!-- Here is a demo: -->

<!-- ![Description](pictures/mast3r_slam_in_ros_demo.gif) -->

## Table of Contents

- [Requirements](#requirements)
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)

---

## Requirements

This package is tested on the following environment configuration:

- Raspberry Pi 5
- Camera module v3
- Raspberry Pi OS
- system python 3.11.2
- venv python 3.11.2
---

## About

- Live-stream input images from camera sensor (i.e. camera module v3)
- Embedded real-time computer vision tasks with mediapipe
- Include multiple tasks e.g. pose estimation, object detection, hand gesture recognition
---

## Installation

Update software for camera module v3:

```bash
sudo apt update
sudo apt full-upgrade # contains software for camera module v3 on RPI OS
```

Now libcamera and picamera2 should be installed in **system python**. Then install other dependencies:

```bash
sudo apt update
sudo apt install python3-opencv -y
sudo apt install python3-venv -y
```

Since installation of **mediapipe** with `$ pip install` is considered as **external installation** on Raspberry Pi 5, it is better install mediapipe in a **virtual env**, which is exposed to the packages installed in **system python**:

```bash
mkdir -p ~/venvs
python3 -m venv --system-site-packages ~/venvs/mediapipe
source ~/venvs/mediapipe/bin/activate
```

Inside venv **mediapipe**, install package `mediapipe`:

```bash
pip install mediapipe
```

Download the pre-trained model for pose estimation:

```bash
mkdir -p ~/ComputerVision_RaspberryPi/checkpoints
wget -O ~/ComputerVision_RaspberryPi/checkpoints/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Usage

Clone this repository:

```bash
cd
git clone https://github.com/MyLovelyAxe/ComputerVision_RaspberryPi.git
```

Download the pre-trained model for pose estimation:

```bash
cd ~/ComputerVision_RaspberryPi
mkdir checkpoints
wget -O checkpoints/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Run **pose estimation**:

```bash
cd ~/ComputerVision_RaspberryPi
python pose_estimation.py
```
