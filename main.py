import os

import cv2

try:
    from deepface import DeepFace
except ImportError as exc:
    raise ImportError(
        "DeepFace is not installed. Install it with 'pip install deepface'."
    ) from exc

FACE_CASCADE_PATH = "model/face_ref.xml"
FACE_MODEL_PATH = "model/face_model.yml"
DATASET_PATH = "dataset"
UNKNOWN_CONFIDENCE_THRESHOLD = 70
AGE_BUCKETS = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100),
]


def create_lbph_recognizer():
    factory = getattr(cv2.face, "LBPHFaceRecognizer_create", None)
    if factory is None:
        raise RuntimeError(
            "OpenCV LBPH recognizer is unavailable. Uninstall 'opencv-python' and "
            "keep only 'opencv-contrib-python', then reinstall dependencies."
        )

    return factory()


def load_names(dataset_path):
    names = {}

    for label_id, person in enumerate(sorted(os.listdir(dataset_path))):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            names[label_id] = person

    return names


def age_to_bucket(age):
    if age is None:
        return "(Unknown Age)"

    for min_age, max_age in AGE_BUCKETS:
        if min_age <= age <= max_age:
            return f"({min_age}-{max_age})"

    if age < AGE_BUCKETS[0][0]:
        min_age, max_age = AGE_BUCKETS[0]
        return f"({min_age}-{max_age})"

    min_age, max_age = AGE_BUCKETS[-1]
    return f"({min_age}-{max_age})"


def predict_age(face_roi):
    if face_roi.size == 0:
        return "(Unknown Age)"

    try:
        result = DeepFace.analyze(
            img_path=face_roi,
            actions=["age"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return "(Unknown Age)"

    if isinstance(result, list):
        result = result[0]

    predicted_age = result.get("age")
    return age_to_bucket(predicted_age)


face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
recognizer = create_lbph_recognizer()
recognizer.read(FACE_MODEL_PATH)
names = load_names(DATASET_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        label, confidence = recognizer.predict(face_gray)
        age_range = predict_age(face_color)

        if confidence < UNKNOWN_CONFIDENCE_THRESHOLD and label in names:
            display_name = names[label]
        else:
            display_name = "Unknown"

        label_text = f"{display_name} {age_range}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
