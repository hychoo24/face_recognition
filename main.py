import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
eyeglasses_ref = cv2.CascadeClassifier("eyeglasses_ref.xml")
camera = cv2.VideoCapture(0) 

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(30, 30), minNeighbors=5)

    return faces

def eyes_detection(roi_gray):
    eyes = eyeglasses_ref.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    return eyes

def drawer_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # uncommand jika ingin menampilkan rectangle pada mata
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        
        # for ex, ey, ew, eh in eyes_detection(roi_gray):
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

def release_resources():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main(): 
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("FaceRecog AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            release_resources()

if __name__ == "__main__":
    main()