import pandas as pd
import librosa
import numpy as np
import os

SAMPLE_RATE = 22050
DURATION = 1
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128

class Preprocessor:

    @staticmethod
    def preprocess_audio(y, sr):

        segments = []
        
        if len(y) < SAMPLES_PER_TRACK:
            
            pad_width = SAMPLES_PER_TRACK - len(y)
            y = np.pad(y, (0, pad_width))
            segments.append(y)
        
        else:
            
            hop = SAMPLES_PER_TRACK // 4
            
            for start in range(0, len(y) - SAMPLES_PER_TRACK + 1, hop):
                end = start + SAMPLES_PER_TRACK
                segment = y[start:end]
                segment = segment.astype(np.float32)
                segments.append(segment)
        
        
        mels = []
        
        for segment in segments:
            
            mel = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_mels=N_MELS,
                hop_length=512,
                n_fft = 1024
            ).astype(np.float32)
            
            log_mel = librosa.power_to_db(mel).astype(np.float32)
            
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
            
            mels.append(log_mel)
        
        return mels