# Twi Interpreter

A two-way speech interpreter for **Asante Twi and English**. You speak Twi, the
other person hears English; flip the direction and the reverse holds. Voice in,
voice out — nothing is uploaded as a file and nothing waits for a recording to
finish.

```
mic ──► VAD segmentation ──► Twi ASR ──► repair + translate ──► TTS ──► speaker
        (webrtcvad)          (Griot      (Claude Opus 5,        (Piper /
                              Edge CTC)   one call)              macOS say)
```

## Why the middle box does two jobs

The Twi recogniser has a word error rate around **0.35** — roughly one word in
three is wrong. A translator handed that output translates noise faithfully. So
repair and translation are a single model call: both need the same evidence
(phonetic shape, conversation context, what would plausibly be said), and
splitting them pays for that reasoning twice while letting the two stages
disagree. One call reconstructs the utterance, renders it in the target
language, and reports how much it had to guess.

Character error rate is much lower than word error rate (0.15 vs 0.35), which
is what makes repair tractable: wrong words are usually phonetically close to
what was said, not unrelated. When they are not recoverable, the stage returns
`low` confidence and the interpreter asks the speaker to repeat instead of
reading out a guess.

## Components

| Stage | What it uses | Quality to expect |
|---|---|---|
| Segmentation | `webrtcvad`, 20 ms frames | 160 ms onset, 700 ms hangover |
| Twi ASR | **Griot Edge** (Qlerqly/GhanaNLP), 61 M-param Conformer CTC, ONNX | CER 0.158 / WER 0.374 causal; CER 0.150 / WER 0.349 with 13-frame lookahead |
| Repair + translation | **Claude Opus 5**, structured output, cached system prompt | — |
| Twi TTS | **Kasanoma** Piper voice | Intelligible, rough prosody; tone is weakest |
| English TTS | macOS `say` (default) or a Piper English voice | Fine |
| English ASR | `faster-whisper` (optional extra) | Fine |

> **Licensing — read before building a product on this.** The Griot *code*
> (`griot-asr-turbo`) is Apache 2.0, but the **published ONNX weights are
> CC BY-NC-SA 4.0** — non-commercial research use with attribution — and the
> `LICENSE` in the weights repo adds: *"Do not publish this artifact until the
> audit is complete."* See `models/griot-edge/LICENSE` and
> `license_audit.json` after fetching. For anything commercial you need either
> a licence from Qlerqly/GhanaNLP or your own model fine-tuned from
> appropriately licensed data.

## Quickstart

```bash
uv venv --python 3.11 .venv
uv pip install -e .                  # add '.[en-asr]' for the English->Twi direction
python scripts/fetch_models.py       # ~300 MB: Griot ASR + Kasanoma Twi voice
cp .env.example .env                 # then set ANTHROPIC_API_KEY (or run `ant auth login`)

twi-interpret                        # Twi in, English out
twi-interpret --direction en-tw      # English in, Twi out
twi-interpret --direction both       # press Enter to flip sides
twi-interpret --file sample.wav      # no microphone needed
```

Useful flags:

| Flag | Effect |
|---|---|
| `--fast` | Turns off model thinking. Lower latency, some accuracy cost on garbled input. |
| `--effort low\|medium\|high` | Repair/translation depth. Default `low`. |
| `--lookahead 0` | Fully causal ASR — lowest latency, WER 0.374 instead of 0.349. |
| `--glossary terms.txt` | Names and domain vocabulary, folded into the cached prompt. |
| `--save-audio corpus/` | Writes each utterance plus its transcript — the seed for fine-tuning. |
| `--list-devices` | Audio device indices for `--input-device` / `--output-device`. |

## Design notes

**Segmentation is the quality knob.** Cut too eagerly and the ASR sees half a
clause, which repair cannot fix because the missing words were never spoken
into the microphone. Cut too late and the listener waits. Twi speakers pause
mid-sentence, hence the relatively long 700 ms hangover (`AudioConfig` in
`src/interpreter/config.py`).

**Feature extraction is the vendor's, not ours.** Griot was trained against a
specific log-mel extractor (400-sample Hann window, 160 hop, 80 Slaney mels,
log floor 1e-5). We import the `onnx_runtime.py` that ships with the weights
rather than reimplementing it, because a plausible-looking reimplementation
that differs in any of those parameters degrades accuracy without raising an
error — the worst failure mode available.

**Prompt caching.** The system prompt (recogniser failure-mode brief, repair
policy, glossary) is one cached block of ~1 000 tokens, above Claude Opus 5's
512-token minimum. Everything that varies per utterance — the transcript and
recent dialogue — goes in the user turn, after the cached prefix.

**Degradation is explicit.** An API error passes the raw transcript through
untranslated and marks it low confidence, so the session continues and the
speaker is prompted to repeat. Server-side refusal fallback (`fallbacks:
"default"`) is enabled, so a safety decline is re-served by Anthropic's
recommended fallback model rather than returning nothing.

## Verification status

`python scripts/selftest.py` runs 15 checks with no microphone, no models, and
no API key — VAD boundaries, request shape, prompt caching, schema strictness,
and every failure path. All pass.

**Not yet verified end to end:** the full mic → speaker loop, Griot weights
loading (the 245 MB download was not performed here), the Kasanoma voice, and a
live API call. Run `scripts/fetch_models.py`, set a key, and try
`twi-interpret --file` on a Twi recording as the first real test.

## What to do next

1. **Measure before improving.** `--save-audio` collects real conversational
   audio with the raw ASR output beside a reviewed reconstruction. That corpus
   is what tells you whether to spend effort on ASR or on TTS.
2. **Cheapest accuracy win: KenLM beam search.** The vendor reports WER 0.269
   with lookahead + KenLM beam versus 0.349 greedy. `ctc_decode.py` is already
   downloaded and `OnnxRuntime.transcribe` takes a `decoder` argument; it needs
   the KenLM model from `Qlerqly/griot-nano-1-kenlm`.
3. **Fine-tune.** Griot for your dialect and domain; Kasanoma from its
   published checkpoints for better Twi prosody and tone.
4. **Then the desktop app.** System-audio capture plus a virtual microphone
   (BlackHole) for Zoom/Meet interpreting — the
   [Cicada](https://github.com/wquguru/cicada) architecture, Apache 2.0, is the
   reference. Build it on this pipeline once the component quality is known.

## Attribution

- Twi ASR: [Griot Edge](https://huggingface.co/Qlerqly/griot-edge) /
  [griot-asr-turbo](https://github.com/GhanaNLP/griot-asr-turbo) — Qlerqly and
  GhanaNLP
- Twi TTS: [Kasanoma](https://github.com/michsethowusu/kasanoma), built on
  [Piper](https://github.com/rhasspy/piper)
- Datasets behind these models:
  [ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
  (2 247 h across 42 Ghanaian languages, 200 h Asante Twi)
