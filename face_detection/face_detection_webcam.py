import cv2
import mediapipe as mp

# Declare Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can not capture frame from webcam.")
            break

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        # Draw boundingbox around the face
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow('Face Detection (Press ESC to exit)', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
