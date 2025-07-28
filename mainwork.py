import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from keras.models import load_model
from music import fetch_songs, add_to_spotify


def modulework():
    st.title("Emotion based music recommendation App")

    st.markdown("---")
    st.markdown("""
    ### Instructions:
    1. Click "Start Detection" to activate the camera
    2. Position your face clearly in the frame
    3. Make facial expressions to detect emotions
    4. Click "Stop Detection" when done
    
    The detected emotion will be stored for further processing.
    """)

    
    # Create session state variables to store the detection results
    if 'emotion' not in st.session_state:
        st.session_state.emotion = None
    
    if 'is_detecting' not in st.session_state:
        st.session_state.is_detecting = False

    if 'songs' not in st.session_state:
        st.session_state.songs = []
        
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    # Load model and labels
    @st.cache_resource
    def load_detection_model():
        model = load_model("model.h5")
        labels = np.load("labels.npy")
        return model, labels
        
    try:
        model, labels = load_detection_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure 'model.h5' and 'labels.npy' are in the same directory as this app")
        return
    
    # Initialize MediaPipe
    @st.cache_resource
    def load_mediapipe():
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        drawing = mp.solutions.drawing_utils
        return holistic, drawing
    
    holistic, drawing = load_mediapipe()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.is_detecting:
            start_button = st.button("Start Detection")
            if start_button:
                st.session_state.is_detecting = True
                st.rerun()
        else:
            stop_button = st.button("Stop Detection")
            if stop_button:
                st.session_state.is_detecting = False
                st.rerun()
    
    # Display detected emotion
    with col2:
        if st.session_state.emotion:
            st.info(f"Detected emotion: {st.session_state.emotion}")
    
    # Camera feed placeholder
    video_placeholder = st.empty()
    
    # Detection logic
    if st.session_state.is_detecting:
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check your camera permissions.")
            st.session_state.is_detecting = False
            return
        
        st.info("Camera active. Processing frames...")
        
        # Process frames for a limited time or until stopped
        start_time = time.time()
        emotion_counts = {}
        
        while st.session_state.is_detecting and (time.time() - start_time < 5):  # 10 second timeout
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break
                
            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = holistic.process(rgb_frame)
            
            # Draw landmarks
            annotated_frame = rgb_frame.copy()
            if results.face_landmarks:
                drawing.draw_landmarks(
                    annotated_frame,
                    results.face_landmarks,
                    mp.solutions.holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                    connection_drawing_spec=drawing.DrawingSpec(color=(0,255,0), thickness=1)
                )
            
            if results.left_hand_landmarks:
                drawing.draw_landmarks(
                    annotated_frame,
                    results.left_hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
                
            if results.right_hand_landmarks:
                drawing.draw_landmarks(
                    annotated_frame,
                    results.right_hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
            
            # Extract features and predict emotion
            if results.face_landmarks:
                # Create feature list as in your original code
                features = []
                
                # Face landmarks
                for i in results.face_landmarks.landmark:
                    features.append(i.x - results.face_landmarks.landmark[1].x)
                    features.append(i.y - results.face_landmarks.landmark[1].y)
                
                # Left hand landmarks
                if results.left_hand_landmarks:
                    for i in results.left_hand_landmarks.landmark:
                        features.append(i.x - results.left_hand_landmarks.landmark[8].x)
                        features.append(i.y - results.left_hand_landmarks.landmark[8].y)
                else:
                    features.extend([0.0] * 42)
                
                # Right hand landmarks
                if results.right_hand_landmarks:
                    for i in results.right_hand_landmarks.landmark:
                        features.append(i.x - results.right_hand_landmarks.landmark[8].x)
                        features.append(i.y - results.right_hand_landmarks.landmark[8].y)
                else:
                    features.extend([0.0] * 42)
                
                # Make prediction
                features = np.array(features).reshape(1, -1)
                prediction = model.predict(features)
                predicted_label = labels[np.argmax(prediction)]
                
                # Count emotions for majority voting
                if predicted_label in emotion_counts:
                    emotion_counts[predicted_label] += 1
                else:
                    emotion_counts[predicted_label] = 1
                
                # Display prediction on frame
                cv2.putText(
                    annotated_frame,
                    f"Emotion: {predicted_label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            # Display the frame
            video_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
            
            # Brief pause to reduce CPU usage
            time.sleep(0.03)
        
        # Release camera
        cap.release()
        
        # Store the most common emotion
        if emotion_counts:
            most_common_emotion = max(emotion_counts, key=emotion_counts.get)
            st.session_state.emotion = most_common_emotion
            st.session_state.is_detecting = False
            st.rerun()
        else:
            st.warning("No emotions detected. Please try again with your face visible.")
            st.session_state.is_detecting = False
    
    # Instructions
   
    additional_context = st.text_area("Add more context about how you're feeling (optional):", 
                                            height=100, key="context_input")
            
    get_recommendations = st.button("Get Song Recommendations")
    if get_recommendations:
                with st.spinner("Fetching personalized song recommendations..."):
                    # Get song recommendations based on emotion and context
                    songs = fetch_songs(st.session_state.emotion, additional_context)
                    st.session_state.songs = songs
                    st.session_state.show_recommendations = True
                    st.rerun()
    if st.session_state.show_recommendations:
            st.write("### Your Personalized Playlist:")
            if st.session_state.songs:
                for i, song in enumerate(st.session_state.songs, 1):
                    st.markdown(f"**{i}.** {song}")
    playlist_name = st.text_input("enter the name of playlist")
    add_ply = st.button("add to spotify")
    if add_ply:
        so = st.session_state.songs
        add_to_spotify(so,playlist_name)

modulework()