import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and cascade
model = load_model("mask_detector.model")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.reshape(face, (1, 224, 224, 3))

            prediction = model.predict(face)
            label = np.argmax(prediction)

            label_text = "No Mask" if label == 1 else "Mask"
            color = (0, 0, 255) if label == 1 else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
