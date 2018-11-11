# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display

ms.use('seaborn-muted')

# Song class
class Song:

    # Initializer / Instance attributes
    def __init__(self, fname):
        self.fname = fname

    # Instance method
    def read_song(self):
        """Read song."""

        # Load song
        y, sr = librosa.load(self.fname, sr=None)

        print(np.shape(y))

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)

        # Make a new figure
        plt.figure(figsize=(12,4))

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Plot params
        plt.colorbar(format='%+02.0f dB')
        plt.title('mel power spectrogram')
        plt.tight_layout()
        plt.show()


