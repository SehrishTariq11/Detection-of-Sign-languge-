import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import socket

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="ASL Detection", page_icon="🖐️", layout="centered")
st.title("🖐️ ASL (American Sign Language) Detection using YOLOv8")

# -----------------------------
# Detect if running on Streamlit Cloud
# -----------------------------
def is_running_on_cloud():
    hostname = socket.gethostname().lower()
    return "streamlit" in hostname or os.getenv("STREAMLIT_RUNTIME") is not None

on_cloud = is_running_on_cloud()

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model_path = "best.pt"

if not os.path.exists(model_path):
    st.warning("⚠️ 'best.pt' not found! Please upload your trained YOLOv8 model.")
    model = None
else:
    model = YOLO(model_path)
    st.success("✅ Model loaded successfully!")

# -----------------------------
# Input Options
# -----------------------------
if on_cloud:
    st.subheader("🎥 Choose Input Type")
    option = st.radio("Select input source:", ["📸 Image Upload"])
    st.info("📹 Live Camera mode is disabled on Streamlit Cloud due to camera access restrictions.")
else:
    st.subheader("🎥 Choose Input Type")
    option = st.radio("Select input source:", ["📸 Image Upload", "🎥 Live Camera"])

# -----------------------------
# Function to extract detected labels
# -----------------------------
def extract_labels(results):
    result = results[0]
    names = result.names
    boxes = result.boxes
    detected_labels = []

    if boxes is not None and len(boxes) > 0:
        for cls in boxes.cls:
            detected_labels.append(names[int(cls)])
    return list(set(detected_labels))  # unique labels

# -----------------------------
# Variable to store final detected text
# -----------------------------
final_prediction = ""

# -----------------------------
# 📸 IMAGE UPLOAD
# -----------------------------
if option == "📸 Image Upload" and model:
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = model(image)
        annotated = results[0].plot()

        st.image(annotated, caption="🔍 Detection Result", use_container_width=True)

        labels = extract_labels(results)
        if labels:
            final_prediction = ", ".join(labels)
        else:
            final_prediction = "No sign detected."

# -----------------------------
# 🎥 LIVE CAMERA (Local Only)
# -----------------------------
elif option == "🎥 Live Camera" and model and not on_cloud:
    st.info("🎦 Use your webcam for live detection.")
    cam_image = st.camera_input("Take a photo")

    if cam_image:
        image = Image.open(cam_image).convert("RGB")
        results = model(image)
        annotated = results[0].plot()
        st.image(annotated, caption="🔍 Detection Result", use_container_width=True)

        labels = extract_labels(results)
        if labels:
            final_prediction = ", ".join(labels)
        else:
            final_prediction = "No sign detected."

# -----------------------------
# 🧾 FINAL DETECTED LETTER BOX
# -----------------------------
st.markdown("---")
st.subheader("🔤 Detected Letter(s)")
st.text_area("Model Prediction:", final_prediction if final_prediction else "No input yet.")
