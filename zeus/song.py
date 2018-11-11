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

        self.read_song()
        self.plot_melspectrogram()

    def info(self, msg):
        print('')
        print('STATUS:')
        print(msg)

    # Instance method
    def read_song(self):
        """Read song."""

        # Load song
        self.info('Loading song...')
        self.y, self.sr = librosa.load(self.fname, sr=None)

        # Get harmonic and percussive components
        self.info('Retrieving harmonic and percussive components...')
        self.y_harmonic, self.y_percussive = librosa.effects.hpss(self.y)

    def plot_melspectrogram(self):
        """Plots the melspectrogram."""

        # Mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)

        # Make a new figure
        plt.figure(figsize=(12,4))

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=self.sr, x_axis='time', y_axis='mel')

        # Plot params
        plt.colorbar(format='%+02.0f dB')
        plt.title('mel power spectrogram')
        plt.tight_layout()
        plt.show()

    def plot_beattracking(self):
        """Plots the song's beat."""

        S = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)

        # We'll use the percussive component for this part
        tempo, beats = librosa.beat.beat_track(y=self.y_percussive, sr=self.sr)

        # Let's re-draw the spectrogram, but this time, overlay the detected beats
        plt.figure(figsize=(12,4))
        librosa.display.specshow(log_S, sr=self.sr, x_axis='time', y_axis='mel')

        # Let's draw transparent lines over the beat frames
        plt.vlines(librosa.frames_to_time(beats),
                1, 0.5 * self.sr,
                colors='w', linestyles='-', linewidth=2, alpha=0.5)

        plt.axis('tight')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()

