import pyaudio
import numpy as np

p = pyaudio.PyAudio()

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate
duration = 1.0   # in seconds
f = 440.0        # sine frequency, Hz

# Generate samples
samples = (volume * np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# Open stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

print("Playing 440 Hz tone...")
stream.write(samples.tobytes())
stream.stop_stream()
stream.close()
p.terminate()
print("Done.")
