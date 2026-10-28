"""Word-level labels for English text, moved onto subword tokens.

The parser labels words; the model reads subword tokens. Each word's labels go on its first
subword, and the rest of the word's subwords get no label. This module has no spaCy, NLTK or
tokenizer imports, so it can be tested anywhere: prepare.py passes in the parsed words, a
WordNet lookup and the tokenizer's offsets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

FUNCTION = "F"  # no WordNet sense applies: function words, numbers, punctuation
UNKNOWN = "none"  # a content word WordNet doesn't list

# spaCy's universal part-of-speech tags -> WordNet's parts of speech.
WORDNET_POS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

Lexnames = Callable[[str, str], set[str]]  # (word, WordNet pos) -> supersense names, e.g. {"noun.food"}


@dataclass(frozen=True)
class Word:
    start: int  # character offset of the word in its line
    lemma: str
    pos: str  # universal part-of-speech tag
    dep: str  # dependency relation


def sense_set(word: Word, lexnames: Lexnames) -> frozenset[str]:
    """Every WordNet supersense the word's lemma can have for its part of speech."""
    pos = WORDNET_POS.get(word.pos)
    if pos is None:
        return frozenset({FUNCTION})
    names = lexnames(word.lemma.lower(), pos)
    return frozenset(names) if names else frozenset({UNKNOWN})


def word_senses(word: str, lexnames: Lexnames) -> frozenset[str]:
    """Every supersense a word form can have under any part of speech; what a dictionary says of it."""
    names: set[str] = set()
    for pos in "nvar":
        names |= lexnames(word.lower(), pos)
    return frozenset(names)


def align(text: str, words: Sequence[Word], offsets: Sequence[tuple[int, int]]) -> list[int | None]:
    """For each subword token, the index of the word it begins, or None if it continues a word.

    A token's position is its first non-space character, so tokens that carry a leading space,
    as byte-level BPE tokens do, still line up with the word that follows the space.
    """
    starts = {w.start: i for i, w in enumerate(words)}
    aligned: list[int | None] = []
    for start, end in offsets:
        first = next((p for p in range(start, end) if not text[p].isspace()), None)
        aligned.append(starts.get(first) if first is not None else None)
    return aligned
