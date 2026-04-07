import streamlit as st
import tempfile
from preprocess import Preprocessor
import librosa
from model import Models
import numpy as np

@st.cache_resource
def load_model():
    return Models.get_model()

model = load_model()

class_map = {
            0 : "air_conditioner",
            1 : "car_horn",
            2 : "children_playing",
            3 : "dog_bark",
            4 : "drilling",
            5 : "engine_idling",
            6 : "gun_shot",
            7 : "jackhammer",
            8 : "siren",
            9 : "street_music"
        }
# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Sound Classification UI",
    page_icon="🔊",
    layout="centered"
)

st.title("🔊 Sound Classification System")

st.write(
    "Upload an audio file and click **Submit** to detect the sound."
)

# -----------------------------------
# FILE UPLOADER
# -----------------------------------

uploaded_file = st.file_uploader(
    label="Upload Audio File",
    type=["wav", "mp3", "ogg"]
)

# -----------------------------------
# SUBMIT BUTTON
# -----------------------------------

submit_button = st.button("Submit")

# -----------------------------------
# PLACEHOLDER FUNCTION
# Replace this with your model logic
# -----------------------------------

def predict_sound(file_path):

    y, sr = librosa.load(file_path, sr=22050)

    mels = Preprocessor.preprocess_audio(y, sr)

    history = {}

    for mel in mels:

        # Fix shape
        mel = mel[..., np.newaxis]
        mel = np.expand_dims(mel, axis=0)

        # Predict
        prob = model.predict(mel, verbose=0)

        pred_idx = np.argmax(prob)

        pred_class = class_map[pred_idx]

        confidence = prob[0][pred_idx]

        # Store confidence
        if pred_class not in history:

            history[pred_class] = [confidence]

        else:

            history[pred_class].append(confidence)

    # Take mean confidence per class
    final_predictions = []
    final_confidences = []

    for key in history:

        mean_conf = np.mean(history[key])

        final_predictions.append(key)

        final_confidences.append(mean_conf)

    return final_predictions, final_confidences


# -----------------------------------
# RESULT DISPLAY
# -----------------------------------

if submit_button:

    if uploaded_file is None:

        st.warning("⚠️ Please upload an audio file first.")

    else:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_audio_path = tmp.name

        st.success("✅ Audio file submitted successfully!")

        # Run prediction
        predicted_classes, confidences = predict_sound(
            temp_audio_path
        )

        # -----------------------------------
        # RESULTS SECTION
        # -----------------------------------

        st.subheader("📊 Results")

        for pred, conf in zip(predicted_classes, confidences):

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Predicted Sound",
                    value=pred
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{conf:.2f}"
                )

        # Optional: Play uploaded audio
        st.subheader("🎧 Uploaded Audio")
        st.audio(uploaded_file)