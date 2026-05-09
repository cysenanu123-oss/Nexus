import sounddevice as sd
import numpy as np

print("Opening Input...")
stream_in = sd.InputStream(channels=1, samplerate=48000)
stream_in.start()

print("Opening Output...")
try:
    stream_out = sd.OutputStream(channels=1, samplerate=48000)
    stream_out.start()
    print("Output successful!")
    stream_out.stop()
except Exception as e:
    print(f"Output failed: {e}")

stream_in.stop()
