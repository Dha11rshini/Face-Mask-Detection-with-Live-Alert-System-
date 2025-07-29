import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detector.model")

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)  # Use 1 or 2 if 0 doesn't work

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to access webcam.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess the face for prediction
        face_resized = cv2.resize(face, (224, 224))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1, 224, 224, 3))

        # Predict mask/no mask
        prediction = model.predict(face_input)
        label = np.argmax(prediction)

        # Set label text and color
        label_text = "No Mask" if label == 1 else "Mask"
        color = (0, 0, 255) if label == 1 else (0, 255, 0)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the result
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
