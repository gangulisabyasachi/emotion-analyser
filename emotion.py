import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 5
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % frame_skip == 0:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = small_frame[y:y + h, x:x + w]

            # Perform emotion analysis
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                emotion = "Unknown"
                print(f"Error analyzing face: {e}")

            # Draw rectangle and label
            cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(small_frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow('Real-time Emotion Detection', small_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
