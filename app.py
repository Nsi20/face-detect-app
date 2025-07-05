# app.py
import streamlit as st
import cv2
import numpy as np
from face_detect.utils import load_emotion_model, detect_faces, detect_emotions, draw_annotations


# Paths
CASCADE_PATH = "haarcascades/haarcascade_frontalface_default.xml"
EMOTION_MODEL_PATH = "emotion_model.h5"

# Load Models
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_model = load_emotion_model(EMOTION_MODEL_PATH)

# Streamlit UI
st.title("🧠 Real-Time Face & Emotion Detection App")
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

if run:
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)
        faces = detect_faces(frame, face_cascade)
        emotions = detect_emotions(frame, faces, emotion_model)
        annotated_frame = draw_annotations(frame, faces, emotions)

        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
else:
    st.write("Click the checkbox to start the webcam.")
