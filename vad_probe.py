import sounddevice as sd, numpy as np, onnxruntime as ort

sess = ort.InferenceSession('jarvis/audio/silero_vad.onnx', providers=['CPUExecutionProvider'])
print('inputs :', [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print('outputs:', [(o.name, o.shape) for o in sess.get_outputs()])

print('3 second normal awaaz mein bolo, abhi...')
a = sd.rec(48000, samplerate=16000, channels=1, dtype='float32')
sd.wait()
x = a[:, 0]
print('peak:', round(float(np.abs(x).max()), 3))

def run(use_context):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros(64, dtype=np.float32)
    probs = []
    for i in range(0, len(x) - 512 + 1, 512):
        chunk = x[i:i+512]
        inp = np.concatenate([ctx, chunk]) if use_context else chunk
        out, state = sess.run(None, {
            'input': inp[None, :].astype(np.float32),
            'state': state,
            'sr': np.array(16000, dtype=np.int64),
        })
        ctx = chunk[-64:]
        probs.append(float(np.squeeze(out)))
    return probs

for flag in (False, True):
    p = run(flag)
    print(f'context={flag}:  max prob={max(p):.3f}   chunks>0.3: {sum(v > 0.3 for v in p)}/{len(p)}')
