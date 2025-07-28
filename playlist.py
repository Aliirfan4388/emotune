import streamlit as st
import time
import cv2
import numpy as np
import mediapipe as mp
import av
from keras.models import load_model

# Load your trained model and labels for emotion detection
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe holistic and hands for landmark detection
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

# Initialize session state variables if not set
if "detect_emotion" not in st.session_state:
    st.session_state.detect_emotion = False
if "start_time" not in st.session_state:
    st.session_state.start_time = 0
if "emotion" not in st.session_state:
    st.session_state.emotion = None

# Define the video processing class for emotion detection
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []
        if res.face_landmarks:
            # Normalize face landmarks relative to landmark index 1
            for lm in res.face_landmarks.landmark:
                lst.append(lm.x - res.face_landmarks.landmark[1].x)
                lst.append(lm.y - res.face_landmarks.landmark[1].y)
            # Process left-hand landmarks (or append zeros)
            if res.left_hand_landmarks:
                for lm in res.left_hand_landmarks.landmark:
                    lst.append(lm.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(lm.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)
            # Process right-hand landmarks (or append zeros)
            if res.right_hand_landmarks:
                for lm in res.right_hand_landmarks.landmark:
                    lst.append(lm.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(lm.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)
            lst = np.array(lst).reshape(1, -1)
            # Predict emotion using the model
            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            st.session_state.emotion = pred  # Save detected emotion in session state
        # Draw landmarks for visualization
        drawing.draw_landmarks(
            frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1)
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Step 1: Show "Detect Emotion" button
if not st.session_state.detect_emotion:
    if st.button("Detect Emotion"):
        st.session_state.detect_emotion = True
        st.session_state.start_time = time.time()

# Step 2: If emotion detection is active, show the video stream for 5 seconds
if st.session_state.detect_emotion:
    elapsed = time.time() - st.session_state.start_time
    if elapsed < 5:
        st.info("Detecting your emotion... Please wait 5 seconds.")
        webrtc_streamer(key="emotion_stream", video_processor_factory=EmotionProcessor)
    else:
        # After 5 seconds, stop the detection and show the result
        if st.session_state.emotion:
            st.success(f"Detected Emotion: {st.session_state.emotion}")
        else:
            st.warning("Could not detect emotion. Please try again.")
        st.session_state.detect_emotion = False  # Reset to allow re-detection if needed

# Step 3: Below, show a text box for extra details/preferences
extra_detail = st.text_area("Describe what you're looking for", "I want uplifting music for a workout")

if st.button("Submit"):
    st.write("Extra Detail Provided:", extra_detail)
    # You can now process extra_detail along with the detected emotion as needed
    # For example, search for songs based on these details
