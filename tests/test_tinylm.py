"""Tests for the NumPy transformer: gradients against finite differences, causality, and learning."""

import unittest

import numpy as np

from hevdicgrm import tinylm as lm


def tiny_setup(dropout):
    cfg = lm.Config(vocab=11, tag_sizes=(5, 3), context=6, width=8, heads=2, layers=2, hidden=12, dropout=dropout)
    rng = np.random.default_rng(0)
    params = lm.init(cfg, rng, dtype=np.float64)
    for name in params:  # move norms and biases off their initial values so their gradients are tested too
        if name.endswith(("_g", "_b")):
            params[name] = params[name] + rng.normal(0, 0.1, params[name].shape)
    x = rng.integers(0, cfg.vocab, (3, 6))
    targets = rng.integers(0, cfg.vocab, (3, 6))
    one_hot = np.eye(5, dtype=bool)[rng.integers(0, 5, (3, 6))]
    several = rng.random((3, 6, 3)) < 0.5
    several[..., 0] |= ~several.any(-1)  # every position needs at least one allowed label
    return cfg, params, x, targets, [one_hot, several]


def loss_and_grads(cfg, params, x, targets, masks, dropout_seed):
    rng = None if dropout_seed is None else np.random.default_rng(dropout_seed)
    logits, tag_logits, cache = lm.forward(params, cfg, x, rng)
    losses, dlogits = lm.nll(logits, targets)
    total = losses.mean()
    dtags = []
    for tl, allowed in zip(tag_logits, masks):
        tag_losses, dtag = lm.set_nll(tl, allowed)
        total += tag_losses.mean()
        dtags.append(dtag)
    return total, dlogits, dtags, cache


class GradientTest(unittest.TestCase):
    def check(self, dropout, dropout_seed):
        cfg, params, x, targets, masks = tiny_setup(dropout)
        _, dlogits, dtags, cache = loss_and_grads(cfg, params, x, targets, masks, dropout_seed)
        grads = lm.backward(params, cfg, cache, dlogits, dtags)
        self.assertEqual(set(grads), set(params))
        rng = np.random.default_rng(1)
        eps = 1e-6
        for name, value in params.items():
            for flat in rng.choice(value.size, size=min(6, value.size), replace=False):
                index = np.unravel_index(flat, value.shape)
                original = value[index]
                value[index] = original + eps
                up = loss_and_grads(cfg, params, x, targets, masks, dropout_seed)[0]
                value[index] = original - eps
                down = loss_and_grads(cfg, params, x, targets, masks, dropout_seed)[0]
                value[index] = original
                numeric = (up - down) / (2 * eps)
                self.assertAlmostEqual(grads[name][index], numeric, delta=1e-6 + 1e-4 * abs(numeric), msg=f"{name}{index}")

    def test_gradients(self):
        self.check(dropout=0.0, dropout_seed=None)

    def test_gradients_with_dropout(self):
        self.check(dropout=0.3, dropout_seed=7)


class LexiconGradientTest(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(3)
        params = {"tok": rng.normal(size=(9, 4)), "lex_w": rng.normal(size=(4, 5)), "lex_b": rng.normal(size=5)}
        allowed = rng.random((9, 5)) < 0.4
        allowed[:, 0] |= ~allowed.any(1)
        _, grads = lm.lexicon_loss(params, allowed)
        eps = 1e-6
        for name, value in params.items():
            for flat in range(value.size):
                index = np.unravel_index(flat, value.shape)
                original = value[index]
                value[index] = original + eps
                up = lm.lexicon_loss(params, allowed)[0]
                value[index] = original - eps
                down = lm.lexicon_loss(params, allowed)[0]
                value[index] = original
                self.assertAlmostEqual(grads[name][index], (up - down) / (2 * eps), delta=1e-7, msg=f"{name}{index}")


class BehaviourTest(unittest.TestCase):
    def test_later_tokens_do_not_change_earlier_predictions(self):
        cfg, params, x, _, _ = tiny_setup(0.0)
        changed = x.copy()
        changed[:, 4:] = (changed[:, 4:] + 1) % cfg.vocab
        before, _, _ = lm.forward(params, cfg, x)
        after, _, _ = lm.forward(params, cfg, changed)
        np.testing.assert_allclose(before[:, :4], after[:, :4], rtol=1e-12)
        self.assertFalse(np.allclose(before[:, 4:], after[:, 4:]))

    def test_one_allowed_label_is_cross_entropy(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(2, 3, 7))
        labels = rng.integers(0, 7, (2, 3))
        a, ga = lm.nll(logits, labels)
        b, gb = lm.set_nll(logits, np.eye(7, dtype=bool)[labels])
        np.testing.assert_allclose(a, b, rtol=1e-12)
        np.testing.assert_allclose(ga, gb, rtol=1e-12, atol=1e-15)

    def test_it_learns_a_repeating_sequence(self):
        cfg = lm.Config(vocab=7, context=12, width=16, heads=2, layers=1, hidden=32, dropout=0.0)
        rng = np.random.default_rng(0)
        params = lm.init(cfg, rng)
        optimizer = lm.AdamW(params, weight_decay=0.0)
        data = np.tile(np.array([3, 1, 4, 1, 5, 2, 6]), 4)[None, :13]
        for _ in range(200):
            logits, _, cache = lm.forward(params, cfg, data[:, :-1])
            losses, dlogits = lm.nll(logits, data[:, 1:])
            optimizer.step(params, lm.backward(params, cfg, cache, dlogits, []), lr=1e-2)
        self.assertLess(-lm.logprobs(params, cfg, data)[:, 7:].mean(), 0.05)  # once a full period is visible


if __name__ == "__main__":
    unittest.main()
