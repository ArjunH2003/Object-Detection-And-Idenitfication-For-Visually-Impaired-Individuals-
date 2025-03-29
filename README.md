# YOLOv5-Based Object Detection and Distance Measurement for the Blind

This project provides real-time object detection, distance measurement, and audio feedback for visually impaired individuals. The system utilizes the YOLOv5 model for detecting objects in live camera feeds and estimates the distance of objects using monocular vision techniques. Critical objects are prioritized for enhanced notifications to ensure safety.

---

## Features
- **Real-Time Object Detection**: Identifies and labels objects in live camera feeds using YOLOv5.
- **Distance Measurement**: Estimates the distance of detected objects using monocular vision methods.
- **Audio Feedback**: Provides audio alerts about detected objects and their distances.
- **Object Prioritization**: Notifies the user about critical objects like "person" or "car" with additional emphasis.
- **Customizability**: Thresholds and priority objects can be modified based on user needs.

---

## Project Structure
- **`detect.py`**: A basic object detection script that uses YOLOv5 for identifying objects.
- **`detect_updated.py`**: An advanced script that includes distance measurement and improved audio feedback.
- **YOLOv5 Repository**: Pre-trained YOLOv5 model and utilities (cloned separately).
- **`requirements.txt`**: Contains dependencies required to run the project.

---

## Prerequisites

### Hardware Requirements
- Raspberry Pi (optional for deployment).
- USB Camera or webcam.
- Headphones or speakers for audio output.

### Software Requirements
- Python 3.8 or higher.
- Virtual environment for isolated Python dependencies.
- Pre-trained YOLOv5 model repository.

---



