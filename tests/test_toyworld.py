"""Tests for the synthetic language: every generated sentence must obey its grammar and meaning rules."""

import random
import unittest

from hevdicgrm.toyworld import ADJECTIVE_CLASSES, NOUN_CLASSES, VERB_CLASSES, World

NOUN_ROLES = {"SUBJ", "OBJ", "RCSUBJ", "RCOBJ", "POBJ"}
WORLD = World(seed=0)


def cls(token):
    return token.sense.split(":")[1]


def features(token):
    return NOUN_CLASSES[cls(token)]


def number(tokens, head):
    return "pl" if head + 1 < len(tokens) and tokens[head + 1].role == "PL" else "sg"


def next_head(tokens, start):
    return next(j for j in range(start + 1, len(tokens)) if tokens[j].role in NOUN_ROLES)


class GeneratedTextTest(unittest.TestCase):
    def check(self, tokens):
        roles = [t.role for t in tokens]
        self.assertEqual(roles[-1], "END")
        for i, t in enumerate(tokens):
            self.assertIn(t.sense, WORLD.senses_of[t.form])
            if t.role == "ADJ":
                self.assertIn(ADJECTIVE_CLASSES[cls(t)], features(tokens[next_head(tokens, i)]))
            if t.role == "DET":
                head_number = number(tokens, next_head(tokens, i))
                self.assertNotEqual((t.form, head_number), ("a", "pl"))
                self.assertNotEqual((t.form, head_number), ("some", "sg"))
        subject, verb = roles.index("SUBJ"), roles.index("VERB")
        subject_feature, object_feature = VERB_CLASSES[cls(tokens[verb])]
        self.assertIn(subject_feature, features(tokens[subject]))
        self.assertEqual(tokens[verb + 1].role == "AGR", number(tokens, subject) == "sg")
        objects = [i for i, r in enumerate(roles) if r == "OBJ"]
        if object_feature:
            self.assertEqual(len(objects), 1)
            self.assertIn(object_feature, features(tokens[objects[0]]))
        else:
            self.assertEqual(objects, [])
        if "REL" in roles:
            clause_verb = roles.index("RCVERB")
            clause_subject_feature, clause_object_feature = VERB_CLASSES[cls(tokens[clause_verb])]
            agrees_singular = tokens[clause_verb + 1].role == "AGR"
            if "RCSUBJ" in roles:  # object relative: the head noun is the clause's object
                clause_subject = roles.index("RCSUBJ")
                self.assertIn(clause_subject_feature, features(tokens[clause_subject]))
                self.assertIn(clause_object_feature, features(tokens[subject]))
                self.assertEqual(agrees_singular, number(tokens, clause_subject) == "sg")
            else:
                self.assertIn(clause_subject_feature, features(tokens[subject]))
                self.assertEqual(agrees_singular, number(tokens, subject) == "sg")
                if clause_object_feature:
                    self.assertIn(clause_object_feature, features(tokens[roles.index("RCOBJ")]))
        if "PREP" in roles:
            needed = "holdable" if tokens[roles.index("PREP")].form == "with" else "place"
            self.assertIn(needed, features(tokens[roles.index("POBJ")]))

    def test_sentences_obey_the_rules(self):
        rng = random.Random(1)
        seen = set()
        for _ in range(3000):
            sentence = WORLD.sentence(rng)
            self.check(sentence)
            seen.update(t.role for t in sentence)
        self.assertEqual(seen, set(WORLD.roles))

    def test_text_ends_at_a_sentence(self):
        text = WORLD.text(500, seed=2)
        self.assertGreaterEqual(len(text), 500)
        self.assertEqual(text[-1].form, ".")
        self.assertEqual(text, WORLD.text(500, seed=2))


class LexiconTest(unittest.TestCase):
    def test_homonyms_pair_two_classes_of_one_part_of_speech(self):
        homonyms = {form: senses for form, senses in WORLD.senses_of.items() if len(senses) > 1}
        nouns = [s for s in homonyms.values() if next(iter(s)).startswith("N:")]
        self.assertEqual(len(nouns), round(0.2 * 100 * len(NOUN_CLASSES) / 2))
        for senses in homonyms.values():
            self.assertEqual(len(senses), 2)
            self.assertEqual(len({s.split(":")[0] for s in senses}), 1)

    def test_vocabulary_and_tags(self):
        self.assertEqual(len(WORLD.vocab), len(set(WORLD.vocab)))
        self.assertEqual(set(WORLD.senses_of), set(WORLD.vocab))
        self.assertEqual(WORLD.senses[0], "F")


class TestItemsTest(unittest.TestCase):
    def test_agreement_items_differ_only_in_agreement(self):
        for right, wrong in WORLD.agreement_items(200, seed=3):
            self.assertEqual(abs(len(right) - len(wrong)), 1)
            self.assertEqual([w for w in right if w != "s"], [w for w in wrong if w != "s"])

    def test_selection_items_cross_two_verbs_and_two_nouns(self):
        items = WORLD.selection_items(200, seed=4)
        self.assertEqual(len(items), 200)
        for item in items:
            (v1, v2), (n1, n2) = item["verbs"], item["nouns"]
            f1 = VERB_CLASSES[next(iter(WORLD.senses_of[v1])).split(":")[1]][1]
            f2 = VERB_CLASSES[next(iter(WORLD.senses_of[v2])).split(":")[1]][1]
            for noun, fits, misfits in ((n1, f1, f2), (n2, f2, f1)):
                classes = [NOUN_CLASSES[s.split(":")[1]] for s in WORLD.senses_of[noun]]
                self.assertTrue(any(fits in c for c in classes))
                self.assertFalse(any(misfits in c for c in classes))
            sentences = item["sentences"]
            self.assertEqual(len({len(s) for s in sentences}), 1)
            self.assertIn(n1, sentences[0])
            self.assertIn(v1, sentences[0])
            self.assertIn(n2, sentences[2])
            self.assertIn(v2, sentences[2])

    def test_sense_items_cue_one_sense_of_a_homonym(self):
        items = WORLD.sense_items(200, seed=5)
        self.assertEqual(len(items), 200)
        for item in items:
            noun = item["noun"]
            x, y = (NOUN_CLASSES[s.split(":")[1]] for s in sorted(WORLD.senses_of[noun]))
            (_, a, *_), _, (_, b, *_), _ = item["sentences"]
            fa = ADJECTIVE_CLASSES[next(iter(WORLD.senses_of[a])).split(":")[1]]
            fb = ADJECTIVE_CLASSES[next(iter(WORLD.senses_of[b])).split(":")[1]]
            # a picks one sense and b the other
            self.assertNotEqual(fa in x, fa in y)
            self.assertNotEqual(fb in x, fb in y)
            self.assertNotEqual(fa in x, fb in x)


if __name__ == "__main__":
    unittest.main()
