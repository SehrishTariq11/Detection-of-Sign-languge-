import streamlit as st
from ultralytics import YOLO
from PIL import Image
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
option = st.radio("Select input source:", ["📸 Image Upload", "🎥 Live Camera"])

# -----------------------------
# Function to extract detected labels
# -----------------------------
def extract_labels(results):
    # Get names and boxes safely from YOLO results
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

        # Get detected class names
        labels = extract_labels(results)
        if labels:
            final_prediction = ", ".join(labels)
        else:
            final_prediction = "No sign detected."

# -----------------------------
# 🎥 LIVE CAMERA
# -----------------------------
elif option == "🎥 Live Camera" and model:
    st.info("🎦 Click 'Start Detection' to begin live sign detection using your webcam.")
    start_button = st.button("▶️ Start Detection")

    if start_button:
        cap = cv2.VideoCapture(0)  # open webcam
        stframe = st.empty()       # placeholder for frames
        label_box = st.empty()     # placeholder for live text

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Cannot access camera.")
                break

            # YOLO prediction
            results = model(frame)
            annotated_frame = results[0].plot()

            # Extract detected labels
            labels = extract_labels(results)
            if labels:
                final_prediction = ", ".join(labels)
            else:
                final_prediction = "No sign detected."

            # Display video and detected text
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
            label_box.markdown(f"### 🔤 **Detected Letter(s):** {final_prediction}")

            # Small delay to control frame rate
            time.sleep(0.05)

            # Stop loop if user presses Stop
            stop = st.button("⏹️ Stop Detection")
            if stop:
                break

        cap.release()
        cv2.destroyAllWindows()


# -----------------------------
# 🧾 FINAL DETECTED LETTER BOX
# -----------------------------
st.markdown("---")
st.subheader("🔤 Detected Letter(s)")
st.text_area("Model Prediction:", final_prediction if final_prediction else "No input yet.")




