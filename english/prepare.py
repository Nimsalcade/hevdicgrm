"""Tag an English corpus for the label experiment: subword tokens with each word's role and senses.

    pip install spacy nltk tokenizers numpy
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('wordnet')"
    python english/prepare.py --train babylm/train_10M/*.train --dev babylm/dev/*.dev --out data/babylm

Writes to --out:
    tokenizer.json   byte-level BPE trained on the training text
    train.npz        per token: ids, the role of the word it begins and the index of that word's
                     sense set (both -1 for a subword that continues a word)
    dev.npz          ids only, for early stopping
    labels.json      the role names, the sense names, the sense sets, and the lexicon: the senses
                     WordNet gives each whole-word token in the vocabulary

Training lines are shuffled in blocks of 100 with a fixed seed, so any prefix of train.npz mixes
all of the corpus's sources and train.py can use the first N tokens as a smaller training set.
"""

from __future__ import annotations

import argparse
import json
import random
from array import array
from pathlib import Path

import numpy as np

import labels

EOS = "<|endoftext|>"


def read_lines(paths: list[str], shuffle_seed: int | None) -> list[str]:
    lines = [" ".join(line.split()) for p in paths for line in Path(p).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if shuffle_seed is None:
        return lines
    blocks = [lines[i : i + 100] for i in range(0, len(lines), 100)]
    random.Random(shuffle_seed).shuffle(blocks)
    return [line for block in blocks for line in block]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab", type=int, default=16_000)
    parser.add_argument("--processes", type=int, default=4, help="spaCy worker processes")
    args = parser.parse_args()

    import spacy
    from nltk.corpus import wordnet
    from tokenizers import ByteLevelBPETokenizer, Tokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    train_lines = read_lines(args.train, shuffle_seed=0)
    dev_lines = read_lines(args.dev, shuffle_seed=None)

    bpe = ByteLevelBPETokenizer(add_prefix_space=True)
    bpe.train_from_iterator(train_lines, vocab_size=args.vocab, min_frequency=2, special_tokens=[EOS])
    bpe.save(str(args.out / "tokenizer.json"))
    tokenizer = Tokenizer.from_file(str(args.out / "tokenizer.json"))
    eos = tokenizer.token_to_id(EOS)

    cache: dict[tuple[str, str], set[str]] = {}

    def lexnames(word: str, pos: str) -> set[str]:
        if (word, pos) not in cache:
            cache[(word, pos)] = {s.lexname() for s in wordnet.synsets(word, pos=pos)}
        return cache[(word, pos)]

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    roles: dict[str, int] = {}
    sets: dict[frozenset[str], int] = {}
    ids, role_ids, set_ids = array("i"), array("i"), array("i")
    for n, (line, doc) in enumerate(zip(train_lines, nlp.pipe(train_lines, batch_size=512, n_process=args.processes))):
        encoding = tokenizer.encode(line)  # one line at a time: a whole corpus of encodings won't fit in memory
        words = [labels.Word(t.idx, t.lemma_, t.pos_, t.dep_) for t in doc if not t.is_space]
        for token, w in zip(encoding.ids, labels.align(line, words, encoding.offsets)):
            ids.append(token)
            if w is None:
                role_ids.append(-1)
                set_ids.append(-1)
            else:
                role_ids.append(roles.setdefault(words[w].dep, len(roles)))
                set_ids.append(sets.setdefault(labels.sense_set(words[w], lexnames), len(sets)))
        ids.append(eos)
        role_ids.append(-1)
        set_ids.append(-1)
        if n % 100_000 == 0:
            print(f"{n:,} of {len(train_lines):,} lines, {len(ids):,} tokens", flush=True)
    np.savez_compressed(args.out / "train.npz", ids=np.frombuffer(ids, np.int32), roles=np.frombuffer(role_ids, np.int32), sets=np.frombuffer(set_ids, np.int32))

    dev = array("i")
    for encoding in tokenizer.encode_batch(dev_lines):
        dev.extend(encoding.ids)
        dev.append(eos)
    np.savez_compressed(args.out / "dev.npz", ids=np.frombuffer(dev, np.int32))

    senses = sorted({s for sense_set in sets for s in sense_set})
    lexicon = {}
    for token, i in tokenizer.get_vocab().items():
        if token.startswith("Ġ") and token[1:].isalpha():  # "Ġ" marks a token that begins a word
            names = labels.word_senses(token[1:], lexnames)
            if names:
                lexicon[i] = sorted(names)
    meta = {
        "eos": eos,
        "vocab": tokenizer.get_vocab_size(),
        "roles": sorted(roles, key=roles.get),
        "senses": senses,
        "sets": [sorted(s) for s in sorted(sets, key=sets.get)],
        "lexicon": lexicon,
    }
    (args.out / "labels.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{len(ids):,} training tokens, {len(roles)} roles, {len(senses)} senses, {len(sets)} sense sets, {len(lexicon):,} lexicon entries")


if __name__ == "__main__":
    main()
