"""A small synthetic language whose every word comes with its sense and its grammatical role.

Real text never says which sense a word is in or what role it plays, so it can't show
cleanly whether those labels help a model learn. This language is generated from them,
so the labels are exact. It has the features the labels are meant to help with:

- Homonyms: some word forms have two senses from different semantic classes, and only
  the context says which one is meant.
- Meaning constraints: verbs and adjectives only combine with nouns whose class has the
  right feature, so "eat" takes food and "clever" takes people or ideas.
- Grammar: number agreement between determiner, noun and verb, including across relative
  clauses with a noun of the other number in between.
- A long tail: words within a class follow a Zipf distribution, so many words are rare.

Plural nouns and singular verbs carry a separate "s" token, as a subword tokenizer would
split English "dogs" and "runs".
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

NOUN_CLASSES: dict[str, frozenset[str]] = {
    "person": frozenset({"animate", "human", "living", "mind", "moving", "physical"}),
    "animal": frozenset({"animate", "living", "moving", "physical"}),
    "food": frozenset({"edible", "perishable", "consumable", "physical"}),
    "drink": frozenset({"drinkable", "consumable", "physical"}),
    "tool": frozenset({"artifact", "holdable", "physical"}),
    "vehicle": frozenset({"artifact", "vehicle", "moving", "physical"}),
    "building": frozenset({"artifact", "place", "physical"}),
    "land": frozenset({"place", "natural", "physical"}),
    "plant": frozenset({"living", "perishable", "natural", "physical"}),
    "idea": frozenset({"abstract", "mind"}),
    "event": frozenset({"abstract", "temporal"}),
}

# Verb class -> (feature its subject needs, feature its object needs, or None if intransitive).
VERB_CLASSES: dict[str, tuple[str, str | None]] = {
    "eat": ("animate", "edible"),
    "drink": ("animate", "drinkable"),
    "hold": ("human", "holdable"),
    "drive": ("human", "vehicle"),
    "build": ("human", "artifact"),
    "see": ("animate", "physical"),
    "discuss": ("human", "abstract"),
    "visit": ("animate", "place"),
    "sleep": ("animate", None),
    "grow": ("living", None),
    "rot": ("perishable", None),
    "break": ("artifact", None),
    "happen": ("temporal", None),
    "matter": ("abstract", None),
}

# Adjective class -> the feature its noun needs.
ADJECTIVE_CLASSES: dict[str, str] = {
    "color": "physical",
    "size": "physical",
    "taste": "consumable",
    "mood": "animate",
    "clever": "mind",
    "fast": "moving",
    "wild": "natural",
    "metal": "artifact",
}

DETERMINERS = ("the", "a", "some")
FUNCTION_WORDS = (".", "s", *DETERMINERS, "that", "in", "with")
NO_SENSE = "F"
ROLES = ("DET", "ADJ", "SUBJ", "PL", "VERB", "AGR", "OBJ", "REL", "RCSUBJ", "RCVERB", "RCOBJ", "PREP", "POBJ", "END")
NUMBERS = ("sg", "pl")


@dataclass(frozen=True)
class Lemma:
    form: str
    pos: str  # "N", "V" or "A"
    cls: str
    rank: int  # 1 is the most frequent word of its class
    weight: float

    @property
    def sense(self) -> str:
        return f"{self.pos}:{self.cls}"


@dataclass(frozen=True)
class Token:
    form: str
    sense: str
    role: str


Sentence = list[Token]


def _other(number: str) -> str:
    return "pl" if number == "sg" else "sg"


class World:
    """The lexicon and grammar of the language, and samplers for text and test items."""

    def __init__(
        self,
        seed: int = 0,
        nouns_per_class: int = 100,
        verbs_per_class: int = 20,
        adjectives_per_class: int = 20,
        noun_homonym_share: float = 0.2,
        other_homonym_share: float = 0.1,
        zipf: float = 1.0,
        relative_rate: float = 0.2,
        phrase_rate: float = 0.2,
    ) -> None:
        rng = random.Random(seed)
        self.relative_rate = relative_rate
        self.phrase_rate = phrase_rate
        specs: list[tuple[str, str, int]] = []
        for pos, classes, per_class in (
            ("N", NOUN_CLASSES, nouns_per_class),
            ("V", VERB_CLASSES, verbs_per_class),
            ("A", ADJECTIVE_CLASSES, adjectives_per_class),
        ):
            specs += [(pos, cls, rank) for cls in classes for rank in range(1, per_class + 1)]
        # Homonyms: pairs of lemmas of the same part of speech, from different classes, share one form.
        # As in natural languages, frequent words are more often ambiguous: lemmas are drawn into
        # pairs in a random order weighted by frequency (Efraimidis and Spirakis's u ** (1 / w) keys).
        partner: dict[int, int] = {}
        for pos, share in (("N", noun_homonym_share), ("V", other_homonym_share), ("A", other_homonym_share)):
            pool = sorted(
                (i for i, s in enumerate(specs) if s[0] == pos), key=lambda i: rng.random() ** (specs[i][2] ** zipf), reverse=True
            )
            pairs = round(share * len(pool) / 2)
            while pairs and len(pool) >= 2:
                first = pool.pop(0)
                match = next((j for j in pool if specs[j][1] != specs[first][1]), None)
                if match is None:
                    break
                pool.remove(match)
                partner[match] = first
                pairs -= 1
        forms = _pseudo_words(rng, len(specs), taken=set(FUNCTION_WORDS))
        form_of = {i: forms[partner.get(i, i)] for i in range(len(specs))}
        self.lemmas = [
            Lemma(form_of[i], pos, cls, rank, 1.0 / rank**zipf) for i, (pos, cls, rank) in enumerate(specs)
        ]
        self.by_class: dict[tuple[str, str], list[Lemma]] = {}
        for lemma in self.lemmas:
            self.by_class.setdefault((lemma.pos, lemma.cls), []).append(lemma)
        self._cum_weights = {key: list(accumulate(lemma.weight for lemma in lemmas)) for key, lemmas in self.by_class.items()}
        self.senses_of: dict[str, frozenset[str]] = {w: frozenset({NO_SENSE}) for w in FUNCTION_WORDS}
        for lemma in self.lemmas:
            self.senses_of[lemma.form] = self.senses_of.get(lemma.form, frozenset()) | {lemma.sense}
        self.vocab = list(FUNCTION_WORDS) + sorted({lemma.form for lemma in self.lemmas})
        self.senses = [NO_SENSE] + sorted({lemma.sense for lemma in self.lemmas})
        self.roles = list(ROLES)

    # Sampling words ------------------------------------------------------------------------

    def _lemma(self, rng: random.Random, pos: str, cls: str) -> Lemma:
        return rng.choices(self.by_class[(pos, cls)], cum_weights=self._cum_weights[(pos, cls)])[0]

    def _noun(self, rng: random.Random, feature: str) -> Lemma:
        classes = [c for c, features in NOUN_CLASSES.items() if feature in features]
        return self._lemma(rng, "N", rng.choice(classes))

    def _adjectives(self, rng: random.Random, noun: Lemma) -> list[Lemma]:
        count = rng.choices((0, 1, 2), weights=(0.6, 0.3, 0.1))[0]
        classes = [c for c, feature in ADJECTIVE_CLASSES.items() if feature in NOUN_CLASSES[noun.cls]]
        return [self._lemma(rng, "A", rng.choice(classes)) for _ in range(count)] if classes else []

    # Building phrases ----------------------------------------------------------------------

    def _np(self, rng: random.Random, noun: Lemma, number: str, role: str, adjectives: bool = True) -> Sentence:
        if number == "sg":
            det = rng.choice(("the", "a"))
        else:
            det = rng.choice(("the", "some"))
        tokens = [Token(det, NO_SENSE, "DET")]
        if adjectives:
            tokens += [Token(a.form, a.sense, "ADJ") for a in self._adjectives(rng, noun)]
        tokens.append(Token(noun.form, noun.sense, role))
        if number == "pl":
            tokens.append(Token("s", NO_SENSE, "PL"))
        return tokens

    @staticmethod
    def _verb(verb: Lemma, number: str, role: str) -> Sentence:
        tokens = [Token(verb.form, verb.sense, role)]
        if number == "sg":
            tokens.append(Token("s", NO_SENSE, "AGR"))
        return tokens

    def _relative(self, rng: random.Random, head: Lemma, number: str) -> Sentence:
        tokens = [Token("that", NO_SENSE, "REL")]
        features = NOUN_CLASSES[head.cls]
        object_verbs = [c for c, (_, obj) in VERB_CLASSES.items() if obj in features]
        if object_verbs and rng.random() < 0.5:
            # Object relative: "the bread that the man eats", where the head is the clause's object.
            verb = self._lemma(rng, "V", rng.choice(object_verbs))
            agent_number = rng.choice(NUMBERS)
            agent = self._noun(rng, VERB_CLASSES[verb.cls][0])
            return tokens + self._np(rng, agent, agent_number, "RCSUBJ") + self._verb(verb, agent_number, "RCVERB")
        verb = self._lemma(rng, "V", rng.choice([c for c, (subj, _) in VERB_CLASSES.items() if subj in features]))
        tokens += self._verb(verb, number, "RCVERB")
        obj = VERB_CLASSES[verb.cls][1]
        if obj:
            tokens += self._np(rng, self._noun(rng, obj), rng.choice(NUMBERS), "RCOBJ")
        return tokens

    def sentence(self, rng: random.Random) -> Sentence:
        verb = self._lemma(rng, "V", rng.choice(list(VERB_CLASSES)))
        subject_feature, object_feature = VERB_CLASSES[verb.cls]
        subject = self._noun(rng, subject_feature)
        number = rng.choice(NUMBERS)
        tokens = self._np(rng, subject, number, "SUBJ")
        if rng.random() < self.relative_rate:
            tokens += self._relative(rng, subject, number)
        tokens += self._verb(verb, number, "VERB")
        if object_feature:
            tokens += self._np(rng, self._noun(rng, object_feature), rng.choice(NUMBERS), "OBJ")
        if rng.random() < self.phrase_rate:
            if object_feature and "human" in NOUN_CLASSES[subject.cls] and rng.random() < 0.5:
                tokens += [Token("with", NO_SENSE, "PREP")] + self._np(rng, self._noun(rng, "holdable"), rng.choice(NUMBERS), "POBJ")
            else:
                tokens += [Token("in", NO_SENSE, "PREP")] + self._np(rng, self._noun(rng, "place"), rng.choice(NUMBERS), "POBJ")
        tokens.append(Token(".", NO_SENSE, "END"))
        return tokens

    def text(self, n_tokens: int, seed: int) -> list[Token]:
        """About `n_tokens` tokens of running text, cut at a sentence end."""
        rng = random.Random(seed)
        tokens: list[Token] = []
        while len(tokens) < n_tokens:
            tokens += self.sentence(rng)
        return tokens

    # Test items ----------------------------------------------------------------------------
    # Each item is a pair or a 2x2 set of sentences that differ in one place. Scoring compares
    # their log-probabilities, so it needs no labels: it works on any model of raw text.

    def agreement_items(self, n: int, seed: int) -> list[tuple[list[str], list[str]]]:
        """(right, wrong) pairs where the verb must agree with a subject across a relative clause
        whose own subject has the other number: "the dog that the cats see runs" vs "... run"."""
        rng = random.Random(seed)
        items = []
        while len(items) < n:
            verb = self._lemma(rng, "V", rng.choice(list(VERB_CLASSES)))
            head = self._noun(rng, VERB_CLASSES[verb.cls][0])
            clause_verbs = [c for c, (_, obj) in VERB_CLASSES.items() if obj in NOUN_CLASSES[head.cls]]
            if not clause_verbs:
                continue
            clause_verb = self._lemma(rng, "V", rng.choice(clause_verbs))
            agent = self._noun(rng, VERB_CLASSES[clause_verb.cls][0])
            number = rng.choice(NUMBERS)
            prefix = (
                self._np(rng, head, number, "SUBJ")
                + [Token("that", NO_SENSE, "REL")]
                + self._np(rng, agent, _other(number), "RCSUBJ")
                + self._verb(clause_verb, _other(number), "RCVERB")
            )
            obj = VERB_CLASSES[verb.cls][1]
            rest = (self._np(rng, self._noun(rng, obj), rng.choice(NUMBERS), "OBJ") if obj else []) + [Token(".", NO_SENSE, "END")]
            right = prefix + self._verb(verb, number, "VERB") + rest
            wrong = prefix + self._verb(verb, _other(number), "VERB") + rest
            items.append(([t.form for t in right], [t.form for t in wrong]))
        return items

    def selection_items(self, n: int, seed: int) -> list[dict]:
        """2x2 sets testing meaning constraints on objects: verbs v1, v2 and nouns n1, n2 where n1 fits
        only v1 and n2 fits only v2. Comparing all four sentences cancels out how common each word is.

        Each item has "sentences" ordered (v1 n1, v1 n2, v2 n2, v2 n1), the first and third valid,
        and "nouns" (n1, n2) for splitting results by how often the nouns were seen in training.
        """
        rng = random.Random(seed)
        transitive = [c for c, (_, obj) in VERB_CLASSES.items() if obj]
        items = []
        while len(items) < n:
            c1, c2 = rng.sample(transitive, 2)
            f1, f2 = VERB_CLASSES[c1][1], VERB_CLASSES[c2][1]
            only1 = [c for c, fs in NOUN_CLASSES.items() if f1 in fs and f2 not in fs]
            only2 = [c for c, fs in NOUN_CLASSES.items() if f2 in fs and f1 not in fs]
            subject_feature = {VERB_CLASSES[c1][0], VERB_CLASSES[c2][0]}
            subjects = [c for c, fs in NOUN_CLASSES.items() if subject_feature <= fs]
            if not (only1 and only2 and subjects):
                continue
            n1, n2 = self._lemma(rng, "N", rng.choice(only1)), self._lemma(rng, "N", rng.choice(only2))
            if not (self._fits_only(n1.form, f1, f2) and self._fits_only(n2.form, f2, f1)):
                continue
            # A homonym verb could make the "invalid" sentences valid under its other sense.
            v1, v2 = self._unambiguous(rng, "V", c1), self._unambiguous(rng, "V", c2)
            if v1 is None or v2 is None:
                continue
            subject = self._lemma(rng, "N", rng.choice(subjects))
            number, object_number = rng.choice(NUMBERS), rng.choice(NUMBERS)
            start = self._np(rng, subject, number, "SUBJ", adjectives=False)
            det = "the"
            plural = ["s"] if object_number == "pl" else []

            def sentence(verb: Lemma, noun: Lemma) -> list[str]:
                return [t.form for t in start + self._verb(verb, number, "VERB")] + [det, noun.form, *plural, "."]

            items.append({
                "sentences": [sentence(v1, n1), sentence(v1, n2), sentence(v2, n2), sentence(v2, n1)],
                "nouns": (n1.form, n2.form),
                "verbs": (v1.form, v2.form),
            })
        return items

    def sense_items(self, n: int, seed: int) -> list[dict]:
        """2x2 sets testing whether context picks a homonym's sense: form f has senses X and Y,
        adjective a fits only X and b only Y, intransitive verb vx fits only X and vy only Y.
        Valid: "the a f vx", "the b f vy". Invalid: "the a f vy", "the b f vx".

        Each item has "sentences" ordered (a vx, a vy, b vy, b vx), the first and third valid.
        """
        rng = random.Random(seed)
        intransitive = [c for c, (_, obj) in VERB_CLASSES.items() if obj is None]
        options = []
        for form, senses in self.senses_of.items():
            nouns = sorted(s.split(":")[1] for s in senses if s.startswith("N:"))
            if len(nouns) != 2:
                continue
            x, y = (NOUN_CLASSES[c] for c in nouns)
            adj_x = [c for c, f in ADJECTIVE_CLASSES.items() if f in x and f not in y]
            adj_y = [c for c, f in ADJECTIVE_CLASSES.items() if f in y and f not in x]
            verb_x = [c for c in intransitive if VERB_CLASSES[c][0] in x and VERB_CLASSES[c][0] not in y]
            verb_y = [c for c in intransitive if VERB_CLASSES[c][0] in y and VERB_CLASSES[c][0] not in x]
            if adj_x and adj_y and verb_x and verb_y:
                options.append((form, adj_x, adj_y, verb_x, verb_y))
        if not options:
            return []
        items = []
        while len(items) < n:
            form, adj_x, adj_y, verb_x, verb_y = rng.choice(options)
            a = self._unambiguous(rng, "A", rng.choice(adj_x))
            b = self._unambiguous(rng, "A", rng.choice(adj_y))
            vx = self._unambiguous(rng, "V", rng.choice(verb_x))
            vy = self._unambiguous(rng, "V", rng.choice(verb_y))
            if None in (a, b, vx, vy):
                continue
            number = rng.choice(NUMBERS)
            noun = [form, "s"] if number == "pl" else [form]
            agree = ["s"] if number == "sg" else []

            def sentence(adj: Lemma, verb: Lemma) -> list[str]:
                return ["the", adj.form, *noun, verb.form, *agree, "."]

            items.append({"sentences": [sentence(a, vx), sentence(a, vy), sentence(b, vy), sentence(b, vx)], "noun": form})
        return items

    def _fits_only(self, form: str, feature: str, other: str) -> bool:
        """True when some sense of `form` has `feature` and none has `other`."""
        classes = [NOUN_CLASSES[s.split(":")[1]] for s in self.senses_of[form] if s.startswith("N:")]
        return any(feature in fs for fs in classes) and not any(other in fs for fs in classes)

    def _unambiguous(self, rng: random.Random, pos: str, cls: str) -> Lemma | None:
        lemmas = [lemma for lemma in self.by_class[(pos, cls)] if len(self.senses_of[lemma.form]) == 1]
        if not lemmas:
            return None
        return rng.choices(lemmas, weights=[lemma.weight for lemma in lemmas])[0]


def _pseudo_words(rng: random.Random, n: int, taken: set[str]) -> list[str]:
    onsets = "b d f g k l m n p r s t v z br dr gr kl pl st tr sk".split()
    vowels = "a e i o u".split()
    codas = ["", "", "", "n", "r", "l", "k", "m"]
    words: list[str] = []
    while len(words) < n:
        word = "".join(rng.choice(onsets) + rng.choice(vowels) for _ in range(rng.choice((2, 2, 3)))) + rng.choice(codas)
        if word not in taken:
            taken.add(word)
            words.append(word)
    return words


def forms(tokens: Sequence[Token]) -> list[str]:
    return [t.form for t in tokens]
