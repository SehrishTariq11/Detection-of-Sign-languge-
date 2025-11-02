import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np
import os
import cv2


# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="ASL Detection", page_icon="🖐️", layout="centered")
st.title("🖐️ ASL (American Sign Language) Detection using YOLOv8")

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
st.subheader("🎥 Choose Input Type")
option = st.radio("Select input source:", ["📸 Image Upload", "🎞️ Video Upload", "🎥 Live Camera"])

# Function to extract detected labels
def extract_labels(results):
    names = results.names
    boxes = results[0].boxes
    detected_labels = [names[int(cls)] for cls in boxes.cls]
    return list(set(detected_labels))  # unique labels

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

        # Get detected class names
        labels = extract_labels(results)
        if labels:
            st.text_area("📝 Predicted Sign(s):", ", ".join(labels))
        else:
            st.info("No sign detected.")

# -----------------------------
# 🎞️ VIDEO UPLOAD
# -----------------------------
elif option == "🎞️ Video Upload" and model:
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        st.video(video_path)
        st.write("🔍 Processing video... please wait...")

        cap = cv2.VideoCapture(video_path)
        output_path = os.path.join(temp_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        all_labels = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # collect labels for summary
            labels = extract_labels(results)
            all_labels.extend(labels)

        cap.release()
        out.release()

        st.success("✅ Detection complete!")
        st.video(output_path)

        # Show unique labels predicted in entire video
        if all_labels:
            unique_labels = list(set(all_labels))
            st.text_area("📝 Predicted Sign(s) in Video:", ", ".join(unique_labels))
        else:
            st.info("No signs detected in this video.")

# -----------------------------
# 🎥 LIVE CAMERA
# -----------------------------
elif option == "🎥 Live Camera" and model:
    st.info("🎦 Capture a photo using your webcam for detection.")
    cam_image = st.camera_input("Take a photo")

    if cam_image:
        image = Image.open(cam_image).convert("RGB")
        results = model(image)
        annotated = results[0].plot()
        st.image(annotated, caption="🔍 Detection Result", use_container_width=True)

        labels = extract_labels(results)
        if labels:
            st.text_area("📝 Predicted Sign(s):", ", ".join(labels))
        else:
            st.info("No sign detected.")



