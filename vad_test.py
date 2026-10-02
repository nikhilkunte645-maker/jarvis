import sounddevice as sd, numpy as np
from jarvis.audio.vad import VADSegmenter

print('Mic device:', sd.query_devices(kind='input')['name'])
print('3 second ke liye zor se bolo, abhi...')
audio = sd.rec(48000, samplerate=16000, channels=1, dtype='int16')
sd.wait()
print('peak (int16):', int(np.abs(audio).max()), 'of 32767')

vad = VADSegmenter()
raw = audio.tobytes()
step = 1024  # 512 samples * 2 bytes
hits = 0
total = 0
for i in range(0, len(raw) - step + 1, step):
    total += 1
    if vad.is_speech(raw[i:i+step]):
        hits += 1
print(f'VAD ne {hits} / {total} chunks ko speech maana')
