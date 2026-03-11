import cv2
import os

face_cascade = cv2.CascadeClassifier("model/face_ref.xml")

name = input("Person's name: ")
path = f"dataset/{name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for x,y,w,h in faces:
        face = gray[y:y+h,x:x+w]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            count += 1
            cv2.imwrite(f"{path}/{count}.jpg", face)
            print("saved", count)

    cv2.imshow("Capture",frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if count >= 30:
        break

cap.release()
cv2.destroyAllWindows()