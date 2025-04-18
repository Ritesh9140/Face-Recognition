# Facial Recognition Attendance System

A Python-based attendance tracker that uses face recognition and blink/liveness detection to monitor student attendance across multiple sessions throughout the day. Students must reappear after a configurable minimum interval to be marked as present.

## Features

- **Session Management**: Automatically divides the day into sessions (9:00–10:10, 10:10–11:10, … up to 20:00).
- **Face Recognition**: Identifies known faces using `face_recognition` and preloaded encodings.
- **Liveness Check**: Implements blink detection via dlib facial landmarks and eye aspect ratio to avoid spoofing.
- **Reappearance Tracking**: Requires students to reappear after a configurable minimum duration (default: 30 minutes) to confirm presence.
- **CSV Logging**: Records attendance data in a daily CSV file (`YYYY-MM-DD.csv`) with fields: Name, Date, Entry Time, Check Time, Status, Branch.
- **Snapshot Capture**: Saves snapshot images (`snapshots/`) for each attendance event, including initial entry and reappearance.

## Prerequisites

- **Python**: 3.x
- **Dependencies**:
  - `face_recognition`
  - `dlib`
  - `imutils`
  - `opencv-python` (cv2)
  - `numpy`
  - `scipy`
- **Model File**: `shape_predictor_68_face_landmarks.dat`
- **Known Faces Directory**: `faces/` containing sample images:
  - `ritesh.jpg`
  - `shashwat.jpg`
  - `venky.jpg`

## Installation

```bash
pip install face_recognition dlib imutils opencv-python numpy scipy
```

Ensure `shape_predictor_68_face_landmarks.dat` and the `faces/` folder are in the same directory as `facialrec.py`.

## Usage

```bash
python facialrec.py [--min-duration <minutes>] [--camera-index <index>]
```

- `--min-duration`: Minimum minutes after entry before reappearance is valid (default: 30).
- `--camera-index`: Index of your webcam device (default: 1).

## How It Works

1. **Load Known Faces**: Encodings are generated at startup from images in `faces/`.
2. **Compute Sessions**: Creates time windows from 9:00 to 20:00 with the first session lasting 1h10m and subsequent sessions lasting 1h each.
3. **Process Each Session**:
   - Continuously capture frames from the webcam.
   - Detect and recognize faces; record entry times when a face is first seen.
   - After `MIN_DURATION` minutes from entry, detect reappearance to mark a student present or mark absent at session end.
4. **Record Attendance**:
   - Append a row to the CSV file for each student at session end.
   - Save a snapshot image under `snapshots/` for each attendance record.

## Directory Structure

```
.
├── facialrec.py
├── faces/
│   ├── ritesh.jpg
│   ├── shashwat.jpg
│   └── venky.jpg
├── shape_predictor_68_face_landmarks.dat
├── snapshots/
└── YYYY-MM-DD.csv
```

## Configuration

- **Blink Sensitivity**: Adjust `EYE_AR_THRESH` and `EYE_AR_CONSEC_FRAMES` in the script for liveness detection.
- **Branch Mapping**: Customize the `get_branch()` function to map names to branches as needed.

## Troubleshooting

- **No Camera Feed**: Try different `--camera-index` values (0, 1, 2, …).
- **Missing Model File**: Download `shape_predictor_68_face_landmarks.dat` from the dlib website and place it alongside the script.
- **Unknown Faces**: Ensure your sample images in `faces/` are clear front-facing photos.

## License

This project is released under the MIT License. Feel free to modify and distribute.

