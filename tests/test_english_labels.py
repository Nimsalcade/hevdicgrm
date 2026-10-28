"""Tests for moving word-level labels onto subword tokens, with a fake WordNet."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "english"))

import labels  # noqa: E402

FAKE_WORDNET = {
    ("bank", "n"): {"noun.group", "noun.object", "noun.artifact"},
    ("bank", "v"): {"verb.possession"},
    ("eat", "v"): {"verb.consumption"},
    ("apple", "n"): {"noun.food", "noun.plant"},
}


def lexnames(word, pos):
    return set(FAKE_WORDNET.get((word, pos), set()))


class SenseSetTest(unittest.TestCase):
    def test_content_words_get_their_supersenses(self):
        self.assertEqual(labels.sense_set(labels.Word(0, "Bank", "NOUN", "nsubj"), lexnames), {"noun.group", "noun.object", "noun.artifact"})
        self.assertEqual(labels.sense_set(labels.Word(0, "eat", "VERB", "ROOT"), lexnames), {"verb.consumption"})

    def test_function_and_unknown_words(self):
        self.assertEqual(labels.sense_set(labels.Word(0, "the", "DET", "det"), lexnames), {labels.FUNCTION})
        self.assertEqual(labels.sense_set(labels.Word(0, "zorp", "NOUN", "dobj"), lexnames), {labels.UNKNOWN})

    def test_word_senses_cover_every_part_of_speech(self):
        self.assertEqual(labels.word_senses("bank", lexnames), {"noun.group", "noun.object", "noun.artifact", "verb.possession"})
        self.assertEqual(labels.word_senses("the", lexnames), set())


class AlignTest(unittest.TestCase):
    TEXT = "The bankers eat apples."
    WORDS = [
        labels.Word(0, "the", "DET", "det"),
        labels.Word(4, "banker", "NOUN", "nsubj"),
        labels.Word(12, "eat", "VERB", "ROOT"),
        labels.Word(16, "apple", "NOUN", "dobj"),
        labels.Word(22, ".", "PUNCT", "punct"),
    ]

    def test_first_subwords_get_the_word_and_the_rest_get_none(self):
        # "The", " bank", "ers", " eat", " app", "les", "." with byte-level offsets that include leading spaces
        offsets = [(0, 3), (3, 8), (8, 11), (11, 15), (15, 19), (19, 22), (22, 23)]
        self.assertEqual(labels.align(self.TEXT, self.WORDS, offsets), [0, 1, None, 2, 3, None, 4])

    def test_offsets_without_leading_spaces(self):
        offsets = [(0, 3), (4, 8), (8, 11), (12, 15), (16, 19), (19, 22), (22, 23)]
        self.assertEqual(labels.align(self.TEXT, self.WORDS, offsets), [0, 1, None, 2, 3, None, 4])

    def test_a_space_only_token_gets_none(self):
        self.assertEqual(labels.align("a  b", [labels.Word(0, "a", "DET", "det"), labels.Word(3, "b", "X", "dep")], [(0, 1), (1, 2), (2, 4)]), [0, None, 1])


if __name__ == "__main__":
    unittest.main()
