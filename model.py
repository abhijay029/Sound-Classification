from tensorflow.keras.models import load_model

class Models:

    @staticmethod
    def get_model():
        
        MODEL_PATH = "Models/sound_detect_1.5M_v5.keras"

        model = load_model(MODEL_PATH)

        return model