# Real-Time Face Recognition

## Overview

This project implements a real-time face recognition system using Python, OpenCV, and the `face_recognition` library. It compares live webcam input with a directory of known face images and identifies individuals in real time.

## Features

- Loads and encodes known face images
- Captures live video from webcam
- Detects and matches faces in real time
- Displays bounding boxes and labels on recognized individuals
- Gracefully handles unknown faces

## How to Run

### 1. Install Dependencies

Make sure you have Python 3.9 installed. Then run:

```bash
pip install -r requirements.txt
```

## Folder Structure

├── known_faces/
│ ├── alice.jpg
│ ├── bob.png
├── face_recognition_realtime.py
├── requirements.txt
├── README.md

Place clear, front-facing images of known individuals inside the `known_faces/` folder. The filename (without extension) is used as the name label.
