import sounddevice as sd
from scipy.io.wavfile import write

# Audio parameters
fs = 44100  # Sample rate (44.1kHz)
seconds = 30  # Duration of recording

print("Recording...")
# Record raw audio (1 channel / mono) without any software filters
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Wait until the recording is finished
print("Finished recording!")

# Save as a standard uncompressed WAV file
write('autumn.wav', fs, myrecording)
