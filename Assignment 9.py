
import cv2
import pytesseract
import streamlit as st
from PIL import Image
from deepface import DeepFace
import time

# ==========================================
# LAB ASSIGNMENT 9
# Cognitive Computing UCS420
# Landing AI - Vision Based Cognitive System
# ==========================================

st.set_page_config(
    page_title="Cognitive Assistant",
    page_icon="🧠"
)

st.title("🧠 Vision Based Cognitive Assistant")
st.write("Two intelligent applications using computer vision.")

option = st.sidebar.selectbox(
    "Choose Application",
    [
        "Medicine Label Reader",
        "Emotion Aware Assistant"
    ]
)

# ==========================================
# 1. MEDICINE LABEL READER
# ==========================================

if option == "Medicine Label Reader":

    st.header("💊 Medicine Label Reader")
    st.write("Upload a medicine bottle image to read its label.")

    st.info("Let's read what's written on your medicine bottle!")

    uploaded_file = st.file_uploader(
        "Upload Medicine Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Medicine Bottle", width="stretch")

        if st.button("Read Label"):

            text = pytesseract.image_to_string(image)

            st.subheader("Extracted Information")

            if text.strip():

                st.text_area(
                    "Medicine Label Text",
                    text,
                    height=200
                )

                lines = [
                    line.strip()
                    for line in text.split("\n")
                    if line.strip()
                ]

                st.subheader("Detected Details")

                if lines:
                    st.write("**Possible Medicine Name:**", lines[0])

                for line in lines:
                    if any(x in line.lower() for x in ["mg", "ml", "mcg"]):
                        st.write("**Possible Dosage:**", line)

                    if any(x in line.lower() for x in ["exp", "expiry"]):
                        st.write("**Possible Expiry Date:**", line)

                    if "batch" in line.lower() or "lot no" in line.lower():
                        st.write("**Batch Information:**", line)

                st.success("Label reading completed!")

                st.warning(
                    "Please verify all medicine information "
                    "with the original packaging or a pharmacist."
                )

            else:
                st.error("No text detected. Try another image.")

# ==========================================
# 2. EMOTION AWARE ASSISTANT
# ==========================================

elif option == "Emotion Aware Assistant":

    st.header("😊 Emotion Aware Cognitive Assistant")

    st.write(
        "This application detects facial expressions "
        "using your webcam."
    )

    st.info("Your expressions make every moment unique!")

    if st.button("Start Emotion Detection"):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Webcam could not be opened.")

        else:

            frame_placeholder = st.empty()
            emotion_placeholder = st.empty()
            stop = st.button("Stop Detection")

            responses = {
                "happy": "Keep smiling! You're looking cheerful.",
                "sad": "Take a little break. Better moments come.",
                "angry": "Take a deep breath. Stay calm.",
                "fear": "Relax. One step at a time.",
                "surprise": "Something unexpected happened!",
                "disgust": "Take a moment to relax.",
                "neutral": "Stay focused and keep going."
            }

            last_emotion = "neutral"
            last_time = 0

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                if time.time() - last_time > 2:

                    try:
                        result = DeepFace.analyze(
                            frame,
                            actions=["emotion"],
                            enforce_detection=False,
                            silent=True
                        )

                        last_emotion = result[0]["dominant_emotion"]

                    except Exception:
                        last_emotion = "neutral"

                    last_time = time.time()

                response = responses.get(
                    last_emotion,
                    "Keep going!"
                )

                cv2.putText(
                    frame,
                    "Emotion: " + last_emotion,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB"
                )

                emotion_placeholder.write(
                    f"**Detected Expression:** {last_emotion.title()}"
                )

                emotion_placeholder.info(response)

                if stop:
                    break

            cap.release()
            cv2.destroyAllWindows()

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption("Lab Assignment 9")
st.caption("Vision-based Cognitive Systems")
