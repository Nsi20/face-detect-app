# utils.py
import cv2
import numpy as np
from keras.models import load_model

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_emotion_model(model_path):
    return load_model(model_path)

def detect_faces(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def detect_emotions(image, faces, model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    predictions = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))  # Match model's input shape
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]
        predictions.append(label)

    return predictions

def draw_annotations(image, faces, emotions):
    for ((x, y, w, h), emotion) in zip(faces, emotions):
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image
