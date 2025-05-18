import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Load YOLOv8 model
model = YOLO("./smakt_scanner/weights/best.pt")  # Thay bằng đường dẫn model của bạn

st.set_page_config(page_title="Giám Sát Giao Thông YOLOv8", layout="wide")

# Tiêu đề
st.markdown("<h1 style='text-align: center; color: black;'>Giám Sát Giao Thông YOLOv8</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Image/Video ")
source_type = st.sidebar.radio("Select Source", ["Image", "Video"])

# =============== IMAGE MODE ================
if source_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        results = model(image)
        res_plotted = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(res_plotted, caption="Detected Image", use_container_width=True)

        with st.expander("Detection Results"):
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"- `{label}`: {conf:.2f}")

    else:
        st.info("📤 Please upload an image from the sidebar.")

# =============== VIDEO MODE ================
elif source_type == "Video":
    video_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        os.remove(video_path)
    else:
        st.info("📥 Please upload a video from the sidebar.")
