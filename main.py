import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime

EMOTION_LABELS = [
    "Neutral", "Happiness", "Surprise", "Sadness",
    "Anger", "Disgust", "Fear", "Contempt"
]

@st.cache_resource
def load_model():
    return ort.InferenceSession("face_detect/emotion-ferplus.onnx")

def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 1, 64, 64)

def detect_emotion(face_img, session):
    input_tensor = preprocess_face(face_img)
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    preds = session.run(None, ort_inputs)[0]
    emotion_idx = np.argmax(preds)
    confidence = preds[0][emotion_idx]
    return EMOTION_LABELS[emotion_idx], float(confidence)

def main():
    st.title("Real-time Face & Emotion Detector")
    run = st.toggle("Start Camera")
    model = load_model()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    FRAME_WINDOW = st.image([])

    cam_index = 0  # Try 0, 1, or 2 depending on your system
    backend = cv2.CAP_DSHOW  # Or try cv2.CAP_MSMF or cv2.CAP_V4L2 if you're on Linux

    cap = cv2.VideoCapture(cam_index, backend)

    if not cap.isOpened():
        st.error("❌ Could not open the webcam.")
        return

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion, confidence = detect_emotion(face_img, model)
            label = f"{emotion} ({confidence*100:.1f}%)"
            color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

if __name__ == "__main__":
    main()

