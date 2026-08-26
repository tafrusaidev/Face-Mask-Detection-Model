import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Set page title and layout
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

# Efficiently load the model and cascade once
@st.cache_resource
def load_resources():
    model = load_model('face_mask_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

try:
    model, face_cascade = load_resources()
    st.title("😷 Face Mask Detection System")
    st.info("Take a photo to check if you are wearing a mask.")

    # Streamlit Camera Input - very fast in browsers
    img_file_buffer = st.camera_input("")

    if img_file_buffer is not None:
        # Convert buffer to OpenCV image
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            st.warning("No face detected!")
        else:
            for (x, y, w, h) in faces:
                # Extract and prepare face for model
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (150, 150)) / 255.0
                face = np.expand_dims(face, axis=0)
                
                # Prediction
                prediction = model.predict(face, verbose=0) # verbose=0 makes it faster/cleaner
                
                if prediction[0][0] < 0.5:
                    label, color = "Mask Detected", (0, 255, 0)
                    st.success(f"✅ {label}")
                else:
                    label, color = "No Mask", (255, 0, 0)
                    st.error(f"❌ {label}")
                
                # Annotate image
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
                cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert BGR to RGB for Streamlit display
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

except Exception as e:
    st.error(f"Error: Ensure 'face_mask_model.h5' is in the project folder. ({e})")