"""Tests for normcheck. A fake client stands in for Jev, so they need neither the SDK nor network access."""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from hevdicgrm import normcheck as nc

REPO = Path(__file__).resolve().parents[1]

BATTERY = {
    "word_column": "word",
    "state_template": "Word: {word}",
    "dimensions": {
        "valence": {"column": "VAL", "instructions": "How positive?", "levels": [str(i) for i in range(1, 10)]},
        "size": {"column": "SIZE", "instructions": "How big?", "levels": [str(i) for i in range(1, 8)]},
    },
}

NORMS = """word,VAL,SIZE
apple,7.0,2.5
war,1.5,6.5
toast (bread),6.5,2.0
toast (speech),5.0,3.5
bank (river),5.5,6.0
bank (money),4.0,5.0
"""


class FakeClient:
    """Answers Score questions with preset expected scores, shaped like `TypeSafeClient.system_one` responses."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def system_one(self, state, questions, model=None):
        word = state.removeprefix("Word: ")
        self.calls.append(word)
        answers = {name: SimpleNamespace(score=self.scores[word][name]) for name in questions}
        return SimpleNamespace(scores=answers, usage=SimpleNamespace(input_tokens=1000), model="jev-fake")


class TempDirTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def write(self, name, text, encoding="utf-8"):
        path = self.dir / name
        path.write_text(text, encoding=encoding)
        return path


class StatisticsTest(unittest.TestCase):
    def test_ties_share_their_average_rank(self):
        self.assertEqual(nc.ranks([3, 1, 3, 2]), [3.5, 1.0, 3.5, 2.0])

    def test_spearman_matches_the_textbook_formula(self):
        # Without ties rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1)); here sum(d^2) = 4, so rho = 0.8.
        self.assertAlmostEqual(nc.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8)

    def test_spearman_extremes(self):
        self.assertAlmostEqual(nc.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(nc.spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertTrue(math.isnan(nc.spearman([1, 2, 3], [5, 5, 5])))

    def test_interval(self):
        # n = 103: z = atanh(0.8) = 1.0986, half-width 1.96 * sqrt(1.06 / 100) = 0.2018.
        low, high = nc.spearman_interval(0.8, 103)
        self.assertAlmostEqual(low, math.tanh(1.0986123 - 0.2017941), places=6)
        self.assertAlmostEqual(high, math.tanh(1.0986123 + 0.2017941), places=6)
        self.assertTrue(all(math.isnan(v) for v in nc.spearman_interval(0.5, 3)))
        self.assertEqual(nc.spearman_interval(1.0, 50), (1.0, 1.0))


class LoadingTest(TempDirTest):
    def test_shipped_batteries_load(self):
        paths = sorted((REPO / "batteries").glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            battery = nc.load_battery(path)
            self.assertTrue(battery.dimensions, path)
            self.assertIn("cat", battery.state("cat"))
            for question in battery.questions().values():
                self.assertEqual(question["type"], "score")
                self.assertGreaterEqual(len(question["criteria"]), 2)

    def test_lowercase_battery_lowercases_the_word(self):
        battery = nc.load_battery(REPO / "batteries" / "lancaster.json")
        self.assertEqual(battery.state("TOAST"), "Word: toast")

    def test_missing_column_names_the_real_headers(self):
        battery = nc.load_battery(self.write("b.json", json.dumps(BATTERY)))
        norms = self.write("n.csv", "word,VAL\napple,7\n")
        with self.assertRaises(ValueError) as caught:
            nc.load_norms(norms, battery)
        self.assertIn("'SIZE'", str(caught.exception))
        self.assertIn("['word', 'VAL']", str(caught.exception))

    def test_blank_and_non_numeric_cells_are_skipped(self):
        battery = nc.load_battery(self.write("b.json", json.dumps(BATTERY)))
        norms = nc.load_norms(self.write("n.csv", "word,VAL,SIZE\napple,,n/a\nwar,1.5,6.5\n", encoding="utf-8-sig"), battery)
        self.assertEqual(norms, {"apple": {}, "war": {"valence": 1.5, "size": 6.5}})

    def test_battery_needs_a_word_slot(self):
        spec = dict(BATTERY, state_template="Word:")
        with self.assertRaises(ValueError):
            nc.load_battery(self.write("b.json", json.dumps(spec)))


class SampleTest(unittest.TestCase):
    WORDS = ["toast", "toast (bread)", "toast (speech)", *(f"w{i}" for i in range(50))]

    def test_senses_of_a_word_stay_together(self):
        toast = {"toast", "toast (bread)", "toast (speech)"}
        for seed in range(30):
            chosen = set(nc.sample(self.WORDS, 10, seed))
            self.assertIn(len(chosen & toast), (0, 3))

    def test_limit_and_seed(self):
        self.assertEqual(nc.sample(self.WORDS, 0, 1), self.WORDS)
        self.assertEqual(nc.sample(self.WORDS, 10, 1), nc.sample(self.WORDS, 10, 1))
        self.assertGreaterEqual(len(nc.sample(self.WORDS, 10, 1)), 10)


class SenseAgreementTest(unittest.TestCase):
    def test_counts_only_clearly_separated_sense_pairs(self):
        rated = {
            "toast (bread)": (5.0, 7.0),  # people: bread > speech by 3 points; Jev agrees
            "toast (speech)": (3.0, 4.0),
            "bank (river)": (2.0, 5.0),  # people differ by 0.2, under the 0.8 gap: not counted
            "bank (money)": (4.0, 5.2),
            "bass (fish)": (2.0, 3.0),  # people: music > fish; Jev reverses them
            "bass (music)": (1.0, 6.0),
            "apple": (4.0, 4.0),  # no sense cue: not counted
        }
        self.assertEqual(nc.sense_agreement(rated, scale_span=8), (2, 0.5))

    def test_no_pairs(self):
        pairs, agreement = nc.sense_agreement({"apple": (1.0, 2.0)}, scale_span=8)
        self.assertEqual(pairs, 0)
        self.assertTrue(math.isnan(agreement))


class CollectTest(TempDirTest):
    def rater(self, calls, fail_on=None, latency=0.0):
        def rate(word):
            calls.append(word)
            time.sleep(latency)
            if word == fail_on:
                raise RuntimeError("API error")
            return {"scores": {"valence": 1.0}, "input_tokens": 10, "model": "jev-fake"}

        return rate

    def test_reruns_reuse_the_cache(self):
        cache, calls = self.dir / "runs" / "c.jsonl", []
        records, new = nc.collect(["a", "b"], self.rater(calls), cache, "f1", workers=2)
        self.assertEqual((set(records), new), ({"a", "b"}, {"a", "b"}))
        records, new = nc.collect(["a", "b", "c"], self.rater(calls), cache, "f1", workers=2)
        self.assertEqual((set(records), new), ({"a", "b", "c"}, {"c"}))
        self.assertEqual(sorted(calls), ["a", "b", "c"])

    def test_changed_questions_are_asked_again(self):
        cache, calls = self.dir / "c.jsonl", []
        nc.collect(["a"], self.rater(calls), cache, "f1", workers=1)
        _, new = nc.collect(["a"], self.rater(calls), cache, "f2", workers=1)
        self.assertEqual(new, {"a"})
        self.assertEqual(calls, ["a", "a"])

    def test_a_line_cut_short_is_skipped_and_ended(self):
        good = json.dumps({"word": "a", "fingerprint": "f1", "scores": {}, "input_tokens": 1, "model": "m"})
        cache, calls = self.write("c.jsonl", good + '\n{"word": "b", "fingerp'), []
        records, new = nc.collect(["a", "b"], self.rater(calls), cache, "f1", workers=1)
        self.assertEqual((set(records), new, calls), ({"a", "b"}, {"b"}, ["b"]))
        lines = cache.read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[-1])["word"], "b")

    def test_an_error_stops_the_run_and_keeps_finished_answers(self):
        # With one worker and a latency like a real request's, w0 is saved before w1 fails, and only
        # w2, already in flight when w1 fails, is sent after it.
        cache, calls = self.dir / "c.jsonl", []
        words = [f"w{i}" for i in range(20)]
        with self.assertRaises(RuntimeError):
            nc.collect(words, self.rater(calls, fail_on="w1", latency=0.1), cache, "f1", workers=1)
        self.assertEqual(calls, ["w0", "w1", "w2"])
        saved = [json.loads(line)["word"] for line in cache.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(saved, ["w0"])


class RunTest(TempDirTest):
    def test_report_from_start_to_finish(self):
        battery = self.write("toy.json", json.dumps(BATTERY))
        norms = self.write("norms.csv", NORMS)
        human = nc.load_norms(norms, nc.load_battery(battery))
        # Jev's scores sit on a 0-based scale but keep the human order, so every rho is 1.
        client = FakeClient({w: {d: v / 2 - 0.5 for d, v in ratings.items()} for w, ratings in human.items()})
        args = nc.parse_args([str(norms), "--battery", str(battery), "--cache", str(self.dir / "c.jsonl"), "--limit", "0"])

        report = nc.run(args, client)
        valence = next(line for line in report.splitlines() if line.startswith("valence"))
        self.assertEqual(valence.split()[1:3], ["6", "1.000"])
        self.assertTrue(valence.endswith("100%"))
        self.assertIn("6 words rated (6 in this run). Input tokens: 6,000, costing $0.0003", report)

        self.assertNotIn("No sense check", report)

        report = nc.run(args, client)
        self.assertIn("(0 in this run)", report)
        self.assertEqual(len(client.calls), 6)

    def test_report_says_when_there_is_no_sense_check(self):
        rows = [nc.Row("touch", 10, 0.5, 0.1, 0.8, 0, math.nan)]
        report = nc.format_report(rows, rated=10, new=10, tokens=100, new_tokens=100)
        self.assertNotIn("same order", report)
        self.assertIn("No sense check", report)


if __name__ == "__main__":
    unittest.main()
