import cv2
import os
import shutil

face_cascade = cv2.CascadeClassifier("model/face_ref.xml")
dataset_root = "dataset"


def get_next_image_index(path):
    max_index = 0

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        if not os.path.isfile(file_path):
            continue

        name, ext = os.path.splitext(file_name)
        if ext.lower() != ".jpg" or not name.isdigit():
            continue

        max_index = max(max_index, int(name))

    return max_index


def choose_dataset_path():
    while True:
        print("Select mode:")
        print("1. Add Dataset")
        print("2. Delete Dataset")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            while True:
                name = input("Enter name: ").strip()
                if name:
                    path = os.path.join(dataset_root, name)
                    os.makedirs(path, exist_ok=True)
                    return path

                print("Name cannot be empty. Try again.")

        if choice == "2":
            while True:
                name = input("Enter the name to delete: ").strip()

                if not name:
                    print("Name cannot be empty. Try again.")
                    continue

                path = os.path.join(dataset_root, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Dataset '{name}' deleted.")
                    return None

                print("Name not found in dataset. Please enter a valid name.")

        print("Invalid choice. Please select 1 or 2.")


path = choose_dataset_path()

if path is None:
    raise SystemExit(0)

cap = cv2.VideoCapture(0)

count = get_next_image_index(path)
session_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengakses webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = gray[y:y + h, x:x + w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s") and len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y + h, x:x + w]
        count += 1
        session_count += 1
        cv2.imwrite(os.path.join(path, f"{count}.jpg"), face)
        print("saved", count)

cap.release()
cv2.destroyAllWindows()
