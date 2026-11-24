"""Combined ASR-repair and translation stage.

Why these are one step and not two: the Twi recogniser gets roughly one word in
three wrong (WER 0.35-0.37), so a translator handed its raw output translates
noise faithfully. Repair needs the same evidence translation needs -- phonetic
shape, conversational context, and which reading actually makes sense -- so
splitting them would mean paying for that reasoning twice and letting the two
stages disagree. One call reconstructs the utterance and renders it in the
target language, and reports how much it had to guess.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Literal

import anthropic

from ..config import TranslateConfig
from . import glossary

LANGUAGE_NAMES = {"tw": "Asante Twi", "en": "English"}

# Documented failure modes of the specific recogniser in front of this stage,
# so the model repairs what actually breaks instead of guessing generically.
ASR_BRIEF = """\
The source text comes from a character-level Conformer CTC recogniser with
greedy decoding. Its known output characteristics:

- Word error rate is about 0.35, so expect roughly one word in three to be
  wrong. Character error rate is about 0.15, so wrong words are usually
  phonetically close to what was said rather than unrelated.
- All lowercase. No punctuation, no capitalisation, no tone marks.
- Twi orthography is produced, including the letters e and o with open-vowel
  forms. Vowel length and nasalisation are frequently wrong.
- Word boundaries are unreliable: words run together, and single words split.
- Code-switched English is transcribed phonetically through a Twi-trained
  vocabulary, so English words can arrive badly deformed.
- Short function words and clitics drop out entirely.
- Proper nouns and loanwords are the least reliable tokens in any utterance.\
"""

SYSTEM_TEMPLATE = """\
You are the interpreting stage of a live two-way speech interpreter for
{source_name} and {target_name}. A person is speaking right now and the other
party is waiting to hear what they said. You receive automatic transcripts, not
text a human typed.

{asr_brief}

Your job, in one pass:

1. Reconstruct the {source_name} utterance the speaker most plausibly produced.
   Use the phonetic shape of the transcript, the conversation so far, and what
   would actually be said in this situation. Prefer the reading that makes the
   utterance coherent over the reading that stays closest to the characters.
2. Render that utterance in natural spoken {target_name}.

How to repair:

- Correct towards what is phonetically reachable from the transcript. A word is
  a candidate if it sounds close to what is written; it is not a candidate just
  because it would fit the topic.
- Rejoin and resplit words freely, and restore dropped function words, when
  doing so yields a well-formed utterance.
- Treat a transcript that is partly English as code-switching and recover the
  English words rather than forcing them into {source_name}.
- Do not add information. No invented names, numbers, places, times, or
  details, and no filling out a fragment into a fuller sentence than was
  spoken. If a stretch of the utterance cannot be recovered, leave that part
  out and lower your confidence instead of inventing a plausible filler.
- Numbers, names, and amounts carry real consequences in this setting. If one
  is not clearly recoverable, say so in `notes` rather than picking the most
  likely value.

How to translate:

- Translate the meaning as a speaker of {target_name} would say it out loud.
  Not word-for-word, and not more formal than the original.
- Preserve the speaker's register, politeness, and whether they are asking,
  telling, or hedging. A question stays a question.
- Keep proper nouns as proper nouns.
- The output is spoken aloud by a synthesiser, so write plain prose: no
  markdown, no bracketed asides, no transliteration guides.

Confidence:

- `high`: the utterance reads cleanly, or repairs were minor and unambiguous.
- `medium`: you resolved real ambiguity and a different reading was possible.
- `low`: substantial parts were unrecoverable, or the meaning turns on a word
  you had to guess. The interpreter asks the speaker to repeat when you say
  `low`, which is the correct outcome for an unintelligible utterance -- do not
  inflate confidence to avoid it.

Set `notes` only when the listener needs to know something about the
reconstruction: what was unclear, or which term you were unsure of. One short
clause, in {target_name}. Leave it empty otherwise.

Do not include internal or system XML tags in your response.

Terms and names that occur in this setting and are likely to be mangled by the
recogniser:

{glossary}\
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "repaired_source": {
            "type": "string",
            "description": "The reconstructed utterance in the source language.",
        },
        "translation": {
            "type": "string",
            "description": "Natural spoken rendering in the target language.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {
            "type": "string",
            "description": "Short caveat for the listener, or an empty string.",
        },
    },
    "required": ["repaired_source", "translation", "confidence", "notes"],
    "additionalProperties": False,
}


@dataclass
class Interpretation:
    repaired_source: str
    translation: str
    confidence: Literal["high", "medium", "low"]
    notes: str = ""
    elapsed: float = 0.0
    degraded: bool = False

    @property
    def needs_repeat(self) -> bool:
        return self.confidence == "low"


class Translator:
    """One Anthropic call per utterance: repair plus translate."""

    def __init__(self, config: TranslateConfig) -> None:
        self.config = config
        # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile, in that order.
        self._client = anthropic.Anthropic()
        self._glossary = glossary.load(config.glossary_path)
        self._history: list[tuple[str, str]] = []

    def _system(self, source: str, target: str) -> list[dict]:
        text = SYSTEM_TEMPLATE.format(
            source_name=LANGUAGE_NAMES[source],
            target_name=LANGUAGE_NAMES[target],
            asr_brief=ASR_BRIEF,
            glossary=self._glossary.strip(),
        )
        # One cache breakpoint on the only stable block. Everything that varies
        # per utterance lives in the user turn, after this prefix.
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def _user_turn(self, raw_text: str, source: str, target: str) -> str:
        parts: list[str] = []
        if self._history:
            recent = self._history[-self.config.context_turns :]
            lines = "\n".join(f"{speaker}: {line}" for speaker, line in recent)
            parts.append(f"Conversation so far:\n{lines}")
        parts.append(
            f"New {LANGUAGE_NAMES[source]} transcript to repair and translate "
            f"into {LANGUAGE_NAMES[target]}:\n{raw_text}"
        )
        return "\n\n".join(parts)

    def interpret(self, raw_text: str, source: str, target: str) -> Interpretation:
        started = time.monotonic()
        raw_text = raw_text.strip()
        if not raw_text:
            return Interpretation("", "", "low", "nothing was transcribed", 0.0, True)

        output_config: dict = {
            "effort": self.config.effort,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        }
        # Thinking is on by default on Claude Opus 5. Adaptive thinking buys
        # accuracy on garbled input; `fast` trades it for latency, which is
        # allowed at effort `high` or below.
        thinking: dict = (
            {"type": "disabled"} if self.config.fast else {"type": "adaptive"}
        )

        try:
            response = self._client.beta.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self._system(source, target),
                messages=[
                    {"role": "user", "content": self._user_turn(raw_text, source, target)}
                ],
                thinking=thinking,
                output_config=output_config,
                # Safety classifiers can decline a request on Claude Opus 5;
                # "default" re-serves it on Anthropic's recommended fallback
                # instead of handing us an empty response.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            print(f"[translate] {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return self._degraded(raw_text, started, "translation unavailable")

        if response.stop_reason == "refusal":
            return self._degraded(raw_text, started, "request declined")

        payload = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return self._degraded(raw_text, started, "unparseable response")

        result = Interpretation(
            repaired_source=data["repaired_source"].strip(),
            translation=data["translation"].strip(),
            confidence=data["confidence"],
            notes=data["notes"].strip(),
            elapsed=time.monotonic() - started,
        )
        self._remember(source, target, result)
        return result

    def _degraded(self, raw_text: str, started: float, note: str) -> Interpretation:
        """Keep the conversation alive when the call fails.

        Passing the raw transcript through untranslated is more useful to the
        other party than silence, as long as it is clearly marked low
        confidence so the interpreter prompts for a repeat.
        """
        return Interpretation(
            repaired_source=raw_text,
            translation=raw_text,
            confidence="low",
            notes=note,
            elapsed=time.monotonic() - started,
            degraded=True,
        )

    def _remember(self, source: str, target: str, result: Interpretation) -> None:
        self._history.append((LANGUAGE_NAMES[source], result.repaired_source))
        self._history.append((LANGUAGE_NAMES[target], result.translation))
        # Keep two turns of slack beyond what we send, so trimming never
        # discards a line the next prompt still wants.
        limit = (self.config.context_turns + 2) * 2
        del self._history[:-limit]

    def reset(self) -> None:
        self._history.clear()
