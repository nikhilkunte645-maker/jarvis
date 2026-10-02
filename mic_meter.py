import sounddevice as sd, numpy as np, time

def cb(indata, frames, t, status):
    peak = float(np.abs(indata).max())
    bar = ('#' * int(min(60, peak * 300))).ljust(60)
    print(bar, round(peak, 3), end='\r')

print('Zor se bolo (Hello Jarvis). Ctrl+C se rokna.')
with sd.InputStream(channels=1, samplerate=16000, blocksize=1600, callback=cb):
    while True:
        time.sleep(0.1)
