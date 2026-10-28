"""Tests for the tagging experiment's bookkeeping: labels, noise, losses, and the report's arithmetic."""

import unittest

import numpy as np

from hevdicgrm import tagexp, tinylm
from hevdicgrm.toyworld import World

WORLD = World(seed=0)
DATA = tagexp.encode(WORLD, WORLD.text(5000, seed=1))
TINY = tagexp.Settings(test_tokens=1500, dev_tokens=800, items=12, width=16, heads=2, layers=1, hidden=32, context=16, max_steps=20)


class LabelTest(unittest.TestCase):
    def test_encoding_keeps_forms_and_senses(self):
        tokens = WORLD.text(200, seed=1)[:200]
        self.assertEqual([WORLD.vocab[i] for i in DATA.ids[:200]], [t.form for t in tokens])
        self.assertEqual([WORLD.senses[i] for i in DATA.senses[:200]], [t.sense for t in tokens])
        for i in range(200):
            allowed = {WORLD.senses[j] for j in np.flatnonzero(DATA.sense_sets[i])}
            self.assertEqual(allowed, WORLD.senses_of[tokens[i].form])

    def test_full_noise_gives_every_ambiguous_token_its_other_sense(self):
        senses, roles = tagexp.corrupt(DATA, tagexp.Arm(senses="gold", sense_noise=1.0, role_noise=1.0), np.random.default_rng(0))
        ambiguous = DATA.sense_sets.sum(1) > 1
        self.assertTrue(ambiguous.any())
        self.assertTrue((senses[ambiguous] != DATA.senses[ambiguous]).all())
        self.assertTrue((senses[~ambiguous] == DATA.senses[~ambiguous]).all())
        self.assertTrue(DATA.sense_sets[np.arange(len(senses)), senses].all())
        self.assertTrue((roles != DATA.roles).all())

    def test_no_noise_changes_nothing(self):
        senses, roles = tagexp.corrupt(DATA, tagexp.Arm(senses="gold"), np.random.default_rng(0))
        np.testing.assert_array_equal(senses, DATA.senses)
        np.testing.assert_array_equal(roles, DATA.roles)


class CoarseTableTest(unittest.TestCase):
    def test_every_class_has_a_coarse_label(self):
        self.assertTrue(set(WORLD.senses) - {"F"} <= set(tagexp.COARSE))

    def test_rows_hold_the_coarse_labels_of_each_forms_senses(self):
        table = tagexp.coarse_table(WORLD)
        self.assertEqual(table.shape, (len(WORLD.vocab), len(tagexp.COARSE_LABELS)))
        for i, form in enumerate(WORLD.vocab):
            expected = {tagexp.COARSE.get(s, s) for s in WORLD.senses_of[form]}
            self.assertEqual({tagexp.COARSE_LABELS[j] for j in np.flatnonzero(table[i])}, expected)


class LossTest(unittest.TestCase):
    def test_every_token_after_the_first_is_scored_once(self):
        cfg = tinylm.Config(vocab=len(WORLD.vocab), context=16, width=16, heads=2, layers=1, hidden=32)
        params = tinylm.init(cfg, np.random.default_rng(0))
        ids = DATA.ids[:50]
        losses = tagexp.token_losses(params, cfg, ids, batch=2)
        self.assertEqual(len(losses), len(ids) - 1)
        # The first window holds tokens 0..16, so it scores tokens 1..16 exactly as logprobs does.
        np.testing.assert_allclose(losses[:16], -tinylm.logprobs(params, cfg, ids[None, :17])[0], rtol=1e-5)


class MultiplierTest(unittest.TestCase):
    PLAIN = [(10, 3.0), (100, 2.0), (1000, 1.5)]

    def test_interpolates_in_log_size(self):
        self.assertEqual(tagexp._multiplier(self.PLAIN, 10, 2.5), "3.16x")  # halfway from 10 to 100 is ~31.6
        self.assertEqual(tagexp._multiplier(self.PLAIN, 100, 2.0), "1.00x")
        self.assertEqual(tagexp._multiplier(self.PLAIN, 100, 2.5), "0.32x")

    def test_bounds_outside_the_measured_sizes(self):
        self.assertEqual(tagexp._multiplier(self.PLAIN, 100, 1.0), ">10.0x")
        self.assertEqual(tagexp._multiplier(self.PLAIN, 100, 3.5), "<0.10x")


class EndToEndTest(unittest.TestCase):
    def test_a_tiny_run_produces_a_full_record(self):
        for arm in tagexp.ARMS:
            record = tagexp.run_one((TINY, arm, 3000, 0, 3000))
            self.assertEqual(record["arm"], arm)
            self.assertLessEqual(record["steps"], TINY.max_steps)
            self.assertTrue(0 < record["test_loss"] < 10)
            for key in ("agreement", "selection", "sense"):
                self.assertTrue(0 <= record[key] <= 1)
            test_text = tagexp.materials(TINY, 3000)["test"]
            self.assertEqual(sum(n for _, n in record["buckets"].values()), len(test_text) - 1)


if __name__ == "__main__":
    unittest.main()
