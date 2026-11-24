"""TTS -> ASR round trip: checks the local chain works, not accuracy.

Run with `python scripts/roundtrip_check.py` after fetching both models.

Synthetic speech is NOT a WER benchmark. Kasanoma's own pronunciation errors
become ASR input errors, so a bad score here can mean either model. It is made
worse by Kasanoma phonemising Twi through espeak's `lfn` (Lingua Franca Nova)
voice, because espeak-ng has no Twi voice -- so the synthetic audio mispronounces
Twi before the recogniser ever hears it. Expect scores well below the vendor's
real-speech figures (CER 0.15 / WER 0.35).

What this does prove: the ONNX graph loads, features line up, CTC decoding emits
Twi orthography, and audio flows end to end without a microphone.
"""
import sys, time
import numpy as np

from interpreter.config import AsrConfig, TtsConfig
from interpreter.asr.griot import GriotRecognizer
from interpreter.tts.piper import PiperSynthesizer

SENTENCES = [
    "Mepɛ sɛ mekɔ fie.",
    "Wo ho te sɛn?",
    "Me din de Kwame.",
    "Mepa wo kyɛw, ka bio.",
    "Mepɛ sɛ mekɔ Kumasi ɔkyena anɔpa.",
]

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

print("loading griot ...", end=" ", flush=True)
t0 = time.monotonic()
asr = GriotRecognizer(AsrConfig())
print(f"{time.monotonic() - t0:.1f}s  (lookahead={asr.lookahead_frames})")

print("loading kasanoma ...", end=" ", flush=True)
t0 = time.monotonic()
tts = PiperSynthesizer(TtsConfig())
print(f"{time.monotonic() - t0:.1f}s")
print()

from scipy.signal import resample_poly
from math import gcd

tot_c = tot_cl = tot_w = tot_wl = 0
for ref in SENTENCES:
    utt = tts.speak(ref)
    audio = utt.audio
    if utt.sample_rate != 16000:
        d = gcd(utt.sample_rate, 16000)
        audio = resample_poly(audio, 16000 // d, utt.sample_rate // d).astype(np.float32)
    out = asr.transcribe(audio)

    r = ref.lower().strip(" .,?!")
    h = out.text.strip()
    cd, wd = edit_distance(r, h), edit_distance(r.split(), h.split())
    tot_c += cd; tot_cl += len(r); tot_w += wd; tot_wl += len(r.split())

    print(f"  ref  {ref}")
    print(f"  tts  {len(audio)/16000:.1f}s @ {utt.sample_rate} Hz, synth {utt.elapsed:.2f}s")
    print(f"  asr  {h!r}   ({out.elapsed:.2f}s, rtf {out.elapsed/(len(audio)/16000):.2f})")
    print(f"       CER {cd/max(len(r),1):.2f}  WER {wd/max(len(r.split()),1):.2f}")
    print()

print(f"aggregate over {len(SENTENCES)} synthetic utterances:")
print(f"  CER {tot_c/max(tot_cl,1):.3f}   WER {tot_w/max(tot_wl,1):.3f}")
print("  (synthetic speech — plumbing check, not a benchmark)")
