# Face Recognition with OpenCV

A simple real-time face recognition project built with Python and OpenCV. This repository covers the full workflow: capturing a face dataset, training an LBPH recognizer, and running live recognition through a webcam.

## Features

- Real-time face detection from webcam input
- Dataset collection per person
- Face recognition model training using `LBPHFaceRecognizer`
- Simple project structure for learning and experimentation

## Project Structure

```text
face-recognition/
├── dataset/
│   ├── person_1/
│   └── person_2/
├── model/
│   ├── face_model.yml
│   ├── face_ref.xml
│   └── eyeglasses_ref.xml
├── capture_dataset.py
├── train_model.py
├── main.py
└── README.md
```

## Requirement

- Python 3.8+
- OpenCV (`opencv-contrib-python`)
- Webcam

## Installation

Install the required packages:

```bash
pip install opencv-contrib-python
```

## Workflow

### 1. Capture Face Dataset

Run the dataset capture script:

```bash
python capture_dataset.py
```

You will be prompted to enter the person's name. A new folder will be created inside `dataset/` using that name.

Controls:

- Press `s` to save a detected face
- Press `q` to quit
- The program stops automatically after 30 captured images

Recommendations for better results:

- Capture 20 to 30 images per person
- Use front-facing and slightly angled poses
- Vary lighting conditions
- Include a few different expressions
- Keep the face sharp, clear, and inside the detection box
- Store only one person per folder

### 2. Train the Recognition Model

Run the training script:

```bash
python train_model.py
```

Expected output:

```bash
Training completed. Saved to model/face_model.yml
```

This step reads all images inside `dataset/` and generates the trained model at `model/face_model.yml`.

### 3. Run Real-Time Recognition

Start the main application:

```bash
python main.py
```

The webcam will open and the system will:

- detect faces in each frame
- predict the face label using the trained LBPH model
- display the recognized name above the detected face

## Notes

- `opencv-contrib-python` is required because the project uses `cv2.face.LBPHFaceRecognizer_create()`.
- If `model/face_model.yml` does not exist, run `train_model.py` first.
- Recognition quality depends heavily on dataset consistency and image quality.

## License

This project is provided for learning and experimentation.
