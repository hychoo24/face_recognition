# Face Detection AI with OpenCV

A simple OpenCV project for real-time face detection using a webcam.

## Features

- Real-time face detection
- Optional eye detection logic in code
- Static label displayed below detected faces

## Project Structure

```text
face_recognition/
├── eyeglasses_ref.xml
├── face_ref.xml
├── main.py
└── README.md
```

## Requirements

- Python 3.8+
- Webcam
- `opencv-python`

## Installation

```bash
pip install opencv-python
```

## Run

```bash
python main.py
```

Press `q` to close the application.

## Notes

- The current implementation performs face detection, not full face recognition.
- XML cascade files must be available in the project root.
- Run the script from the project root so file paths resolve correctly.
