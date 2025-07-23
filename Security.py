import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the trained MobileNet model
model = load_model("mobilenet_human_detection.h5", compile=False)

# Streamlit UI Design
st.set_page_config(page_title="AI Security System", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #007bff;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #555;
        }
        .status-box {
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        .human {
            background-color: #28a745;
            color: white;
        }
        .non-human {
            background-color: #dc3545;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='main-title'>🚨 AI-Powered Security System</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Detect motion and classify objects in real time!</p>", unsafe_allow_html=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def classify_object(image):
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    return "Human" if prediction < 0.5 else "Non-Human"

frame_placeholder = st.empty()
status_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame. Exiting...")
        break

    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))

    label = "No Motion Detected"
    color_class = ""

    if boxes:
        x_min = min([box[0] for box in boxes])
        y_min = min([box[1] for box in boxes])
        x_max = max([box[2] for box in boxes])
        y_max = max([box[3] for box in boxes])
        detected_object = frame[y_min:y_max, x_min:x_max]

        if detected_object.size > 0:
            label = classify_object(detected_object)
            color_class = "human" if label == "Human" else "non-human"
            color = (0, 255, 0) if label == "Human" else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "Human":
                os.system("say 'Intruder detected'")

    status_placeholder.markdown(f"<div class='status-box {color_class}'>{label}</div>", unsafe_allow_html=True)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")
