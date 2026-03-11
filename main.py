import cv2
import os

face_cascade = cv2.CascadeClassifier("model/face_ref.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/face_model.yml")

dataset_path = "dataset"

names = {}
label_id = 0

for person in os.listdir(dataset_path):
    names[label_id] = person
    label_id += 1

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for x,y,w,h in faces:

        face = gray[y:y+h,x:x+w]

        label, confidence = recognizer.predict(face)

        name = names[label]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        release_resources()

cap.release()
cv2.destroyAllWindows()

def release_resources():
    camera.release()
    cv2.destroyAllWindows()
    exit()