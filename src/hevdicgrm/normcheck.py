"""Check Jev's word ratings against human rating norms.

This is the Meaning Atlas's first test. Before paying to ask Jev hundreds of
questions about every dictionary sense, measure how closely its answers track
human ratings that already exist. Each word gets one request that asks every
dimension of a battery as a Score question on the scale the human raters used.

The report gives, per dimension, the Spearman correlation between Jev's expected
score and the human mean rating. For norms that rate ambiguous words one sense
at a time, written like "toast (bread)", it also gives how often Jev orders two
senses of the same word the way people did. That second number decides whether
a per-sense Atlas is worth paying for: a model can rank words well while
ignoring which sense it was asked about.

    pip install -e .
    export TYPESAFE_API_KEY=...
    normcheck glasgow.csv --battery batteries/glasgow.json --limit 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typesafe_sdk import TypeSafeClient

PRICE_PER_MILLION_INPUT_TOKENS = 0.042  # USD; Jev's output tokens are free.

# A sense-cued entry such as "toast (bread)": a word, then a cue that picks one of its senses.
SENSE_CUE = re.compile(r"^(?P<word>.*\S)\s*\((?P<cue>[^()]+)\)$")

# Two senses count as rated differently only when their human means differ by at least this
# share of the scale; smaller gaps are within rater noise and would pull agreement toward 50%.
MIN_SENSE_GAP = 0.1


@dataclass(frozen=True)
class Dimension:
    name: str
    column: str
    instructions: str
    levels: tuple[str, ...]


@dataclass(frozen=True)
class Battery:
    word_column: str
    state_template: str
    dimensions: tuple[Dimension, ...]
    lowercase: bool = False

    def state(self, word: str) -> str:
        return self.state_template.replace("{word}", word.lower() if self.lowercase else word)

    def questions(self) -> dict[str, dict[str, Any]]:
        return {
            d.name: {"type": "score", "instructions": d.instructions, "criteria": list(d.levels)}
            for d in self.dimensions
        }

    def fingerprint(self, model: str | None) -> str:
        """Identify the exact requests sent, so a rerun never reuses answers to different wording."""
        spec = {"state": self.state_template, "lowercase": self.lowercase, "questions": self.questions(), "model": model}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def load_battery(path: Path) -> Battery:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "{word}" not in spec["state_template"]:
        raise ValueError(f"{path}: state_template must contain {{word}}")
    dimensions = []
    for name, d in spec["dimensions"].items():
        if len(d["levels"]) < 2:
            raise ValueError(f"{path}: dimension {name!r} needs at least two levels")
        dimensions.append(Dimension(name, d["column"], d["instructions"], tuple(d["levels"])))
    return Battery(spec["word_column"], spec["state_template"], tuple(dimensions), spec.get("lowercase", False))


def load_norms(path: Path, battery: Battery, word_column: str | None = None) -> dict[str, dict[str, float]]:
    """Read each word's human mean rating per dimension, skipping blank or non-numeric cells."""
    word_column = word_column or battery.word_column
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in (word_column, *(d.column for d in battery.dimensions)) if c not in header]
        if missing:
            raise ValueError(f"{path} has no column(s) {missing}; its columns are {header}")
        norms = {}
        for row in reader:
            word = (row[word_column] or "").strip()
            if word:
                norms[word] = {d.name: value for d in battery.dimensions if (value := _number(row[d.column])) is not None}
    return norms


def _number(cell: str | None) -> float | None:
    try:
        value = float(cell)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def base_word(entry: str) -> str:
    match = SENSE_CUE.match(entry)
    return match["word"] if match else entry


def sample(words: Sequence[str], limit: int, seed: int) -> list[str]:
    """Pick about `limit` words at random, keeping each word's sense-cued entries together; 0 keeps all.

    Norm files are usually alphabetical, so taking the first rows would be a biased sample.
    """
    if limit <= 0 or limit >= len(words):
        return list(words)
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        groups[base_word(word)].append(word)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    chosen: list[str] = []
    for key in keys:
        if len(chosen) >= limit:
            break
        chosen.extend(groups[key])
    return chosen


Rater = Callable[[str], dict[str, Any]]


def jev_rater(client: TypeSafeClient, battery: Battery, model: str | None) -> Rater:
    """Rate a word with one request that asks every dimension of the battery about it."""
    questions = battery.questions()

    def rate(word: str) -> dict[str, Any]:
        response = client.system_one(state=battery.state(word), questions=questions, model=model)
        return {
            "scores": {name: answer.score for name, answer in response.scores.items()},
            "input_tokens": response.usage.input_tokens or 0,
            "model": response.model,
        }

    return rate


def collect(
    words: Sequence[str], rate: Rater, cache: Path, fingerprint: str, workers: int
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Rate every word the cache lacks, appending each answer as it arrives so an interrupted run resumes.

    Returns the records for `words` and the set of words rated by this call.
    """
    text = cache.read_text(encoding="utf-8") if cache.exists() else ""
    records = {}
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut short when an earlier run was killed
            continue
        if record.get("fingerprint") == fingerprint:
            records[record["word"]] = record
    todo = [w for w in words if w not in records]
    cache.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        with cache.open("a", encoding="utf-8") as out:
            if text and not text.endswith("\n"):
                out.write("\n")
            futures = {pool.submit(rate, word): word for word in todo}
            for done, future in enumerate(as_completed(futures), 1):
                word = futures[future]
                record = {"word": word, "fingerprint": fingerprint, **future.result()}
                out.write(json.dumps(record) + "\n")
                out.flush()
                records[word] = record
                if done % 100 == 0 or done == len(todo):
                    print(f"rated {done}/{len(todo)} words", file=sys.stderr)
    finally:
        # After an error, drop the queued words instead of sending them; the cache keeps what finished.
        pool.shutdown(cancel_futures=True)
    return {w: records[w] for w in words if w in records}, set(todo)


def ranks(values: Sequence[float]) -> list[float]:
    """Rank from 1, giving tied values the average of the ranks they span."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            result[i] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else math.nan


def spearman_interval(rho: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval from Fisher's z, using the 1.06 / (n - 3) variance Fieller et al. (1957) give for Spearman's rho."""
    if n < 4 or math.isnan(rho):
        return math.nan, math.nan
    if abs(rho) == 1:
        return rho, rho
    half = z * math.sqrt(1.06 / (n - 3))
    return math.tanh(math.atanh(rho) - half), math.tanh(math.atanh(rho) + half)


@dataclass(frozen=True)
class Row:
    dimension: str
    n: int
    rho: float
    low: float
    high: float
    sense_pairs: int
    sense_agreement: float


def evaluate(
    records: Mapping[str, Mapping[str, Any]], norms: Mapping[str, Mapping[str, float]], battery: Battery
) -> list[Row]:
    rows = []
    for d in battery.dimensions:
        rated = {
            word: (record["scores"][d.name], norms[word][d.name])
            for word, record in records.items()
            if d.name in record["scores"] and d.name in norms.get(word, {})
        }
        jev = [j for j, _ in rated.values()]
        human = [h for _, h in rated.values()]
        rho = spearman(jev, human) if len(rated) >= 3 else math.nan
        # Battery levels mirror the human scale one point per level, so it spans len(levels) - 1 points.
        pairs, agreement = sense_agreement(rated, len(d.levels) - 1)
        rows.append(Row(d.name, len(rated), rho, *spearman_interval(rho, len(rated)), pairs, agreement))
    return rows


def sense_agreement(rated: Mapping[str, tuple[float, float]], scale_span: float) -> tuple[int, float]:
    """Count same-word sense pairs people rated clearly apart, and the share Jev puts in the same order.

    `rated` maps entries to (Jev score, human mean). Chance agreement is 50%; a Jev tie counts as disagreement.
    """
    senses: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for entry, scores in rated.items():
        if match := SENSE_CUE.match(entry):
            senses[match["word"]].append(scores)
    pairs = agree = 0
    for group in senses.values():
        for (jev_a, human_a), (jev_b, human_b) in combinations(group, 2):
            if abs(human_a - human_b) >= MIN_SENSE_GAP * scale_span:
                pairs += 1
                agree += (jev_a - jev_b) * (human_a - human_b) > 0
    return pairs, agree / pairs if pairs else math.nan


def format_report(rows: Sequence[Row], rated: int, new: int, tokens: int, new_tokens: int) -> str:
    senses = any(r.sense_pairs for r in rows)
    header = f"{'dimension':<20}{'words':>7}{'rho':>8}  {'95% CI':<16}"
    if senses:
        header += f"{'sense pairs':>12}{'same order':>12}"
    lines = [header]
    for r in rows:
        interval = "-" if math.isnan(r.low) else f"[{r.low:.2f}, {r.high:.2f}]"
        line = f"{r.dimension:<20}{r.n:>7}{_format(r.rho, '.3f'):>8}  {interval:<16}"
        if senses:
            line += f"{r.sense_pairs:>12}{_format(r.sense_agreement, '.0%'):>12}"
        lines.append(line)
    lines.append("")
    if not senses:
        lines.append("No sense check: no two rated entries are senses of one word written like 'toast (bread)'.")
    lines.append(
        f"{rated} words rated ({new} in this run). Input tokens: {tokens:,}, "
        f"costing ${_cost(tokens):.4f} (${_cost(new_tokens):.4f} in this run)."
    )
    return "\n".join(lines)


def _format(value: float, spec: str) -> str:
    return "-" if math.isnan(value) else format(value, spec)


def _cost(tokens: int) -> float:
    return tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="normcheck", description="Compare Jev's word ratings with human rating norms.")
    parser.add_argument("norms", type=Path, help="CSV of human mean ratings, one row per word")
    parser.add_argument("--battery", type=Path, required=True, help="JSON file of questions; see batteries/")
    parser.add_argument("--word-column", help="override the battery's word column")
    parser.add_argument(
        "--limit", type=int, default=1000, help="rate about this many words, keeping senses together; 0 rates all (default 1000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for picking the sample (default 0)")
    parser.add_argument("--cache", type=Path, help="JSONL of Jev's answers, reused on reruns (default runs/<battery>.jsonl)")
    parser.add_argument("--workers", type=int, default=8, help="requests in flight at once (default 8)")
    parser.add_argument("--model", help="Jev model (default: the SDK's, jev-latest unless TYPESAFE_DEFAULT_MODEL is set)")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, client: TypeSafeClient) -> str:
    battery = load_battery(args.battery)
    norms = load_norms(args.norms, battery, args.word_column)
    words = sample(list(norms), args.limit, args.seed)
    cache = args.cache or Path("runs") / f"{args.battery.stem}.jsonl"
    records, new = collect(words, jev_rater(client, battery, args.model), cache, battery.fingerprint(args.model), args.workers)
    tokens = sum(r["input_tokens"] for r in records.values())
    new_tokens = sum(records[w]["input_tokens"] for w in new)
    return format_report(evaluate(records, norms, battery), len(records), len(new), tokens, new_tokens)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Imported here so the rest of this module, and its tests, run without the SDK installed.
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    # A run sends many requests, so ride out rate limits for longer than the SDK's default.
    with TypeSafeClient(retry=RetryPolicy(max_retries=6, timeout=120.0)) as client:
        print(run(args, client))
    return 0


if __name__ == "__main__":
    sys.exit(main())
