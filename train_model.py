import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
names = {}

label_id = 0


def create_lbph_recognizer():
    factory = getattr(cv2.face, "LBPHFaceRecognizer_create", None)
    if factory is None:
        raise RuntimeError(
            "OpenCV LBPH recognizer is unavailable. Uninstall 'opencv-python' and "
            "keep only 'opencv-contrib-python', then reinstall dependencies."
        )

    return factory()

for person in os.listdir(dataset_path):
    names[label_id] = person
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        faces.append(img)
        labels.append(label_id)

    label_id += 1

labels = np.array(labels)

recognizer = create_lbph_recognizer()

recognizer.train(faces, labels)

recognizer.save("model/face_model.yml")

print("Training completed. Saved to model/face_model.yml")
