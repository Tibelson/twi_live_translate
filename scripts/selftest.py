#!/usr/bin/env python3
"""Offline checks that need no microphone, no models, and no API key.

Covers the two things most likely to break silently: VAD segmentation
boundaries, and the shape of the request sent to the translation stage.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-selftest-placeholder")

from interpreter.audio.vad import SegmentChunker  # noqa: E402
from interpreter.config import AudioConfig, TranslateConfig  # noqa: E402
from interpreter.translate.llm import RESPONSE_SCHEMA, Translator  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


def test_segmentation() -> None:
    print("segmentation")
    config = AudioConfig()
    chunker = SegmentChunker(config)
    frame_samples = chunker.frame_samples
    rng = np.random.default_rng(0)

    def noisy(ms: int) -> list[np.ndarray]:
        return [
            rng.normal(0, 0.25, frame_samples).astype(np.float32)
            for _ in range(ms // config.frame_ms)
        ]

    def silent(ms: int) -> list[np.ndarray]:
        return [np.zeros(frame_samples, dtype=np.float32) for _ in range(ms // config.frame_ms)]

    frames = (
        silent(300) + noisy(1200) + silent(1000) + noisy(800) + silent(1000)
        + noisy(60) + silent(1000)  # 60 ms blip must be rejected
    )
    segments = list(chunker.segments(iter(frames)))
    durations = [round(len(s) / config.sample_rate, 2) for s in segments]
    check("two utterances found", len(segments) == 2, f"durations {durations}")
    check("sub-minimum blip rejected", all(d > 0.4 for d in durations))


def test_translate_request() -> None:
    print("translate stage")
    captured: dict = {}

    class Block:
        type = "text"

        def __init__(self, text: str) -> None:
            self.text = text

    class Response:
        stop_reason = "end_turn"

        def __init__(self, payload: dict) -> None:
            self.content = [Block(json.dumps(payload))]

    def create(**kwargs):  # noqa: ANN003
        captured.clear()
        captured.update(kwargs)
        return Response(
            {
                "repaired_source": "Mepɛ sɛ mekɔ Kumasi ɔkyena anɔpa.",
                "translation": "I want to go to Kumasi tomorrow morning.",
                "confidence": "medium",
                "notes": "",
            }
        )

    translator = Translator(TranslateConfig())
    translator._client = types.SimpleNamespace(
        beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    )

    result = translator.interpret("mepe se meko kumase kyena anopa", "tw", "en")
    check("translation returned", result.translation.startswith("I want to go"))
    check("confidence parsed", result.confidence == "medium")
    check("not flagged degraded", result.degraded is False)
    check("thinking is adaptive by default", captured["thinking"] == {"type": "adaptive"})
    check("structured output requested", captured["output_config"]["format"]["type"] == "json_schema")
    check("refusal fallback enabled", captured["fallbacks"] == "default")
    check(
        "system prompt has one cache breakpoint",
        len(captured["system"]) == 1
        and captured["system"][0]["cache_control"] == {"type": "ephemeral"},
    )
    check(
        "system prompt exceeds the 512-token cache minimum",
        len(captured["system"][0]["text"]) > 2048,
        f"{len(captured['system'][0]['text'])} chars",
    )
    check("schema is strict", RESPONSE_SCHEMA["additionalProperties"] is False)
    check(
        "every schema property is required",
        set(RESPONSE_SCHEMA["required"]) == set(RESPONSE_SCHEMA["properties"]),
    )

    translator.interpret("aane me nim", "tw", "en")
    check(
        "second turn carries conversation context",
        captured["messages"][0]["content"].startswith("Conversation so far:"),
    )


def test_degradation() -> None:
    print("failure handling")
    import anthropic

    translator = Translator(TranslateConfig())

    class Refused:
        stop_reason = "refusal"
        content: list = []

    translator._client = types.SimpleNamespace(
        beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **k: Refused()))
    )
    refused = translator.interpret("mepe se meko", "tw", "en")
    check("refusal degrades to low confidence", refused.needs_repeat and refused.degraded)

    def boom(**kwargs):  # noqa: ANN003
        raise anthropic.APIConnectionError(request=None)

    translator._client.beta.messages.create = boom
    broken = translator.interpret("mepe se meko", "tw", "en")
    check("api error passes transcript through", broken.translation == "mepe se meko")
    check("api error marked low confidence", broken.needs_repeat and broken.degraded)


def main() -> int:
    test_segmentation()
    print()
    test_translate_request()
    print()
    test_degradation()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
