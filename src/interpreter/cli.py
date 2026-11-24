"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.text import Text

from .config import Config
from .pipeline import Interpreter, Turn, read_wav

FLAGS = {"tw": "Twi", "en": "English"}
CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twi-interpret",
        description="Two-way Asante Twi <-> English speech interpreter.",
    )
    parser.add_argument(
        "--direction",
        choices=["tw-en", "en-tw", "both"],
        default="tw-en",
        help="tw-en: speak Twi, hear English. both: press Enter to flip sides.",
    )
    parser.add_argument("--file", type=Path, help="interpret a wav file instead of the mic")
    parser.add_argument("--no-speak", action="store_true", help="text output only")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="turn off model thinking for lower latency, at some accuracy cost",
    )
    parser.add_argument(
        "--effort", choices=["low", "medium", "high"], help="repair/translation effort"
    )
    parser.add_argument("--glossary", type=Path, help="term list for the cached prompt")
    parser.add_argument(
        "--save-audio",
        type=Path,
        help="write each utterance plus its transcript here (fine-tuning corpus)",
    )
    parser.add_argument("--input-device", type=int, help="input device index")
    parser.add_argument("--output-device", type=int, help="output device index")
    parser.add_argument(
        "--list-devices", action="store_true", help="print audio devices and exit"
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        metavar="FRAMES",
        help="ASR lookahead frames: 0 is fully causal, 13 trades latency for accuracy",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    config.direction = args.direction
    config.save_audio = args.save_audio
    if args.input_device is not None:
        config.audio.input_device = args.input_device
    if args.output_device is not None:
        config.audio.output_device = args.output_device
    if args.lookahead is not None:
        config.asr.lookahead_frames = args.lookahead
    if args.effort is not None:
        config.translate.effort = args.effort
    if args.fast:
        config.translate.fast = True
    if args.glossary is not None:
        config.translate.glossary_path = args.glossary
    return config


def render_turn(console: Console, turn: Turn) -> None:
    interpretation = turn.interpretation
    style = CONFIDENCE_STYLE[interpretation.confidence]

    header = Text()
    header.append(f"{FLAGS[turn.source]} -> {FLAGS[turn.target]}", style="bold cyan")
    header.append(f"  {turn.audio_seconds:.1f}s audio", style="dim")
    console.print(header)

    console.print(Text(f"  asr      {turn.transcript.text or '(nothing)'}", style="dim"))
    if interpretation.repaired_source != turn.transcript.text:
        console.print(Text(f"  repaired {interpretation.repaired_source}", style="white"))
    console.print(Text(f"  -> {interpretation.translation}", style=f"bold {style}"))

    footer = Text("     ", style="dim")
    footer.append(f"{interpretation.confidence}", style=style)
    if interpretation.notes:
        footer.append(f" · {interpretation.notes}", style="dim italic")
    if interpretation.degraded:
        footer.append(" · untranslated passthrough", style="red")
    footer.append(
        f"  ·  asr {turn.transcript.elapsed:.2f}s"
        f" · interpret {interpretation.elapsed:.2f}s"
        f" · tts {turn.tts_seconds:.2f}s"
        f" · total {turn.total_seconds:.2f}s",
        style="dim",
    )
    console.print(footer)
    console.print()


def watch_for_flip(interpreter: Interpreter, console: Console) -> None:
    """Blank line on stdin flips the active direction."""
    for _ in sys.stdin:
        source, target = interpreter.flip()
        console.print(
            Text(f"  ↻ now {FLAGS[source]} -> {FLAGS[target]}", style="bold magenta")
        )


def main() -> int:
    args = build_parser().parse_args()
    console = Console()

    if args.list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        return 0

    config = config_from_args(args)
    try:
        interpreter = Interpreter(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if args.file is not None:
        audio = read_wav(args.file, config.audio.sample_rate)
        render_turn(console, interpreter.handle(audio, speak=not args.no_speak))
        return 0

    source, target = interpreter.current_direction()
    console.print(
        Text(f"Interpreting {FLAGS[source]} -> {FLAGS[target]}.", style="bold")
    )
    if args.direction == "both":
        console.print(Text("Press Enter to flip direction. Ctrl-C to stop.", style="dim"))
        threading.Thread(
            target=watch_for_flip, args=(interpreter, console), daemon=True
        ).start()
    else:
        console.print(Text("Speak when ready. Ctrl-C to stop.", style="dim"))
    console.print()

    try:
        interpreter.run(on_turn=lambda turn: render_turn(console, turn))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
