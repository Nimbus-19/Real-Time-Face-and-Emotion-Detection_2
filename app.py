import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pickle
import tempfile

# Page config
st.set_page_config(
    page_title="Real-Time Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# --- Custom Style ---
st.markdown("""
<style>
.main {
    padding: 1rem;
}
.upload-text {
    font-size: 1.1rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("emotion logo.png", width=100)
with col2:
    st.title("Real-Time Face & Emotion Detection")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "Detect emotions from images or webcam using deep learning. "
    "Uses face detection (OpenCV), feature extraction (OpenFace), and "
    "emotion recognition (ML model)."
)

# --- Load Models ---
@st.cache_resource
def load_ml_models():
    models_dir = os.path.join('5_DjangoApp', 'facerecognition', 'static', 'models')

    # Face detection
    face_detector_model = cv2.dnn.readNetFromCaffe(
        os.path.join(models_dir, 'deploy.prototxt.txt'),
        os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    )

    # Feature extractor
    face_feature_model = cv2.dnn.readNetFromTorch(
        os.path.join(models_dir, 'openface.nn4.small2.v1.t7')
    )

    # Emotion recognition
    emotion_recognition_model = pickle.load(
        open(os.path.join(models_dir, 'machinelearning_face_emotion.pkl'), 'rb')
    )

    return face_detector_model, face_feature_model, emotion_recognition_model


# --- Process Image Function ---
def process_image(image, face_model, feature_model, emotion_model):
    """Detect faces and predict emotions."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (h, w) = image_cv.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    detections = face_model.forward()

    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Boundary check
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            face_roi = image_cv[startY:endY, startX:endX]

            if face_roi.size == 0:
                continue

            # Feature extraction
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96),
                                              (0, 0, 0), swapRB=True, crop=False)
            feature_model.setInput(face_blob)
            vec = feature_model.forward()

            # Emotion prediction
            try:
                pred = emotion_model.predict(vec)[0]
            except Exception as e:
                pred = "Unknown"

            # Draw results
            text = f"{pred} ({confidence:.2f})"
            cv2.rectangle(image_cv, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image_cv, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            results.append({
                'face_detect_score': float(confidence),
                'emotion': pred,
            })

    return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), results


# --- Load Models ---
try:
    face_detector, feature_model, emotion_model = load_ml_models()
    st.success("✅ ML models loaded successfully!")
except Exception as e:
    st.error(f"Error loading ML models: {str(e)}")
    st.stop()


# --- Mode Selection ---
mode = st.radio("Choose Input Mode:", ["Upload Image", "Use Webcam"])

# --- Image Upload Mode ---
if mode == "Upload Image":
    st.markdown('<p class="upload-text">Upload an image file</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        processed_image, results = process_image(input_image, face_detector, feature_model, emotion_model)

        st.image(processed_image, caption='Processed Image with Emotions', use_column_width=True)

        if results:
            st.subheader("Detection Results:")
            for i, res in enumerate(results, 1):
                st.write(f"**Face {i}:** Emotion - `{res['emotion']}`")
        else:
            st.warning("No faces detected.")


# --- Webcam Mode ---
elif mode == "Use Webcam":
    st.info("Click the checkbox below to start webcam and detect emotions in real-time.")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        processed_image, results = process_image(frame_pil, face_detector, feature_model, emotion_model)

        FRAME_WINDOW.image(processed_image)

    camera.release()
    st.write("Webcam stopped.")
