# Footfall-Counter
# Footfall Counter — YOLOv8 (Real-time)

## Overview
Real-time footfall counter that detects people using YOLOv8, tracks them with a centroid tracker, and counts entries/exits when people cross a configurable horizontal counting line.

## Features
- YOLOv8 person detection
- Simple centroid-based tracking with stable ID assignment
- Entry/exit counting when objects cross a virtual horizontal line
- Live visualization (bounding boxes, IDs, trajectories, counters, FPS)
- Works with webcam, saved videos, or RTSP streams

## Requirements
- Python 3.8+
- pip packages:
  - ultralytics
  - opencv-python
  - numpy

Install dependencies:
```bash
pip install ultralytics opencv-python numpy
