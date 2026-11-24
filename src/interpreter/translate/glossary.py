"""Optional term list folded into the cached system prompt.

Names, places, and domain vocabulary are exactly what a character-level CTC
model mangles worst, and exactly what context cannot recover. Listing them up
front is the cheapest accuracy win available.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_GLOSSARY = """\
Kumasi, Accra, Kejetia, Adum, Tech (KNUST), Manhyia
Asantehene, Otumfuo, Nana, Agya, Maame, Owura, Awuraa
trotro, mobile money / momo, chop bar, waakye, kenkey, fufu, banku
"""


def load(path: Path | None) -> str:
    if path is None:
        return DEFAULT_GLOSSARY
    text = path.read_text(encoding="utf-8").strip()
    return text or DEFAULT_GLOSSARY
