#!/usr/bin/env python3
"""Download the Twi ASR model and the Twi TTS voice into ./models.

Griot Edge ships its own reference ONNX runtime (log-mel extraction + chunked
inference + greedy CTC). We download that module alongside the weights and call
it directly rather than reimplementing feature extraction, because any mismatch
with the training-time extractor degrades accuracy silently.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

GRIOT_REPO = "Qlerqly/griot-edge-onnx"
GRIOT_FILES = [
    "model.onnx",
    "config.json",
    "vocab.json",
    "onnx_runtime.py",  # vendor's feature extraction + CTC decode
    "ctc_decode.py",  # optional KenLM beam search hook
    "LICENSE",
    "metrics.json",
]
KASANOMA_TWI_ZIP = (
    "https://github.com/michsethowusu/kasanoma/releases/download/v1/kasanoma_model-twi.zip"
)

ROOT = Path(__file__).resolve().parent.parent


def fetch_griot(dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for name in GRIOT_FILES:
        print(f"  {name} ...", end=" ", flush=True)
        try:
            path = hf_hub_download(repo_id=GRIOT_REPO, filename=name)
        except Exception as exc:  # optional files may not exist
            print(f"skipped ({exc.__class__.__name__})")
            continue
        shutil.copy(path, dest / name)
        print("ok")


def fetch_kasanoma(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {KASANOMA_TWI_ZIP} ...", end=" ", flush=True)
    with urllib.request.urlopen(KASANOMA_TWI_ZIP) as response:
        blob = response.read()
    print(f"{len(blob) / 1e6:.0f} MB")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            if member.endswith((".onnx", ".json")) and not member.startswith("__"):
                target = dest / Path(member).name
                target.write_bytes(archive.read(member))
                print(f"  extracted {target.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-only", action="store_true", help="skip the Twi TTS voice")
    parser.add_argument("--tts-only", action="store_true", help="skip the Twi ASR model")
    args = parser.parse_args()

    if not args.tts_only:
        print(f"Griot Edge ASR -> models/griot-edge  ({GRIOT_REPO})")
        fetch_griot(ROOT / "models" / "griot-edge")
        print(
            "\n  NOTE: these weights are CC BY-NC-SA 4.0 (non-commercial research use,\n"
            "  attribution required). See models/griot-edge/LICENSE before shipping.\n"
        )
    if not args.asr_only:
        print("Kasanoma Twi TTS -> models/kasanoma-twi")
        fetch_kasanoma(ROOT / "models" / "kasanoma-twi")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
