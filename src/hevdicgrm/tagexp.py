"""Does labelling every word with its sense and role let a small model learn a language from less text?

Every arm trains the same small transformer on the same synthetic text (see toyworld). The arms
differ only in what else the model must predict at each token:

    plain                nothing else
    roles                the token's grammatical role, as a parser would supply
    sense-sets           every sense the token's form can have, as a dictionary alone would supply
    senses               the token's sense in context, as a dictionary plus a perfect Jev would supply
    roles+sense-sets     both of the cheap ones: no Jev needed
    roles+senses         everything
    roles+senses-noisy   as roles+senses, with 30% of ambiguous tokens given the wrong sense and
                         10% of roles wrong, as an imperfect Jev and parser would
    random-classes       a control: each word form gets a fixed random class, as many classes as
                         there are senses. It regularizes like a sense head but carries no meaning,
                         so any gain the sense arms show beyond it comes from what the labels say.
    lexicon              no labels in the text; instead, on every step each vocabulary word's
                         embedding must predict that word's possible senses. This is the dictionary
                         used as a table, and it covers words the training text never shows.
    roles+sense-sets+lexicon   everything that needs no Jev
    random-lexicon       a control for lexicon: a fixed random class per word

All arms are tested on raw text alone, so labels can only help through what the model learned from
them. The report compares held-out loss, accuracy on targeted test items, and how much more text the
plain model needs to match each arm.

    python -m hevdicgrm.tagexp run --out results/tagexp.jsonl
    python -m hevdicgrm.tagexp report results/tagexp.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np

from hevdicgrm import tinylm
from hevdicgrm.toyworld import Token, World


@dataclass(frozen=True)
class Arm:
    roles: bool = False
    senses: str | None = None  # "set", "gold", or "random" for a meaningless class per form
    sense_noise: float = 0.0  # share of ambiguous tokens given their other sense
    role_noise: float = 0.0  # share of tokens given a random wrong role
    lexicon: str | None = None  # "sets" to teach embeddings every word's senses, "random" for the control


ARMS = {
    "plain": Arm(),
    "roles": Arm(roles=True),
    "sense-sets": Arm(senses="set"),
    "senses": Arm(senses="gold"),
    "roles+sense-sets": Arm(roles=True, senses="set"),
    "roles+senses": Arm(roles=True, senses="gold"),
    "roles+senses-noisy": Arm(roles=True, senses="gold", sense_noise=0.3, role_noise=0.1),
    "random-classes": Arm(senses="random"),
    "lexicon": Arm(lexicon="sets"),
    "roles+sense-sets+lexicon": Arm(roles=True, senses="set", lexicon="sets"),
    "random-lexicon": Arm(lexicon="random"),
}

SIZES = (10_000, 20_000, 40_000, 80_000, 200_000)
BUCKETS = ((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, None))  # training occurrences of the target word
RARE = 10  # a word seen at most this often in training counts as rare


@dataclass(frozen=True)
class Settings:
    world_seed: int = 0
    test_tokens: int = 100_000
    dev_tokens: int = 30_000
    items: int = 600
    width: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 256
    context: int = 64
    dropout: float = 0.1
    batch: int = 32
    lr: float = 2e-3
    warmup: int = 100
    eval_every: int = 100
    patience: int = 4
    max_steps: int = 8000


@dataclass
class Encoded:
    ids: np.ndarray  # (n,) token ids
    senses: np.ndarray  # (n,) sense in context
    sense_sets: np.ndarray  # (n, senses) every sense the form can have
    roles: np.ndarray  # (n,)


def lexicon_table(world: World) -> np.ndarray:
    """(vocab, senses) mask of every sense each word form can have: the dictionary."""
    word = {w: i for i, w in enumerate(world.vocab)}
    sense = {s: i for i, s in enumerate(world.senses)}
    table = np.zeros((len(world.vocab), len(world.senses)), dtype=bool)
    for form, senses in world.senses_of.items():
        table[word[form], [sense[s] for s in senses]] = True
    return table


def encode(world: World, tokens: list[Token]) -> Encoded:
    word = {w: i for i, w in enumerate(world.vocab)}
    sense = {s: i for i, s in enumerate(world.senses)}
    role = {r: i for i, r in enumerate(world.roles)}
    ids = np.array([word[t.form] for t in tokens], dtype=np.int32)
    return Encoded(
        ids=ids,
        senses=np.array([sense[t.sense] for t in tokens], dtype=np.int16),
        sense_sets=lexicon_table(world)[ids],
        roles=np.array([role[t.role] for t in tokens], dtype=np.int16),
    )


def corrupt(data: Encoded, arm: Arm, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The sense and role labels an imperfect tagger would produce."""
    senses, roles = data.senses.copy(), data.roles.copy()
    ambiguous = np.flatnonzero(data.sense_sets.sum(1) > 1)
    flip = ambiguous[rng.random(len(ambiguous)) < arm.sense_noise]
    for i in flip:  # every ambiguous form has exactly two senses: take the other one
        senses[i] = np.flatnonzero(data.sense_sets[i] & (np.arange(data.sense_sets.shape[1]) != senses[i]))[0]
    wrong = np.flatnonzero(rng.random(len(roles)) < arm.role_noise)
    n_roles = int(roles.max()) + 1
    roles[wrong] = (roles[wrong] + rng.integers(1, n_roles, len(wrong))) % n_roles
    return senses, roles


@cache
def materials(settings: Settings, max_size: int) -> dict:
    """The language, its training text, and the held-out text and test items every run shares."""
    world = World(seed=settings.world_seed)
    return {
        "world": world,
        "train": encode(world, world.text(max_size, seed=1)),
        "dev": encode(world, world.text(settings.dev_tokens, seed=2)).ids,
        "test": encode(world, world.text(settings.test_tokens, seed=3)).ids,
        "agreement": world.agreement_items(settings.items, seed=4),
        "selection": world.selection_items(settings.items, seed=5),
        "sense": world.sense_items(settings.items, seed=6),
    }


def config(settings: Settings, world: World, arm: Arm) -> tinylm.Config:
    tag_sizes = ((len(world.roles),) if arm.roles else ()) + ((len(world.senses),) if arm.senses else ())
    return tinylm.Config(
        vocab=len(world.vocab),
        tag_sizes=tag_sizes,
        context=settings.context,
        width=settings.width,
        heads=settings.heads,
        layers=settings.layers,
        hidden=settings.hidden,
        dropout=settings.dropout,
    )


def train(
    settings: Settings, cfg: tinylm.Config, data: Encoded, dev: np.ndarray, arm: Arm, seed: int, table: np.ndarray | None = None
) -> tuple[tinylm.Params, dict]:
    """Train until held-out loss stops improving; return the best parameters seen.

    `table` is the dictionary (vocab x senses) the lexicon arms learn from.
    """
    rng = np.random.default_rng(seed)
    params = tinylm.init(cfg, rng)
    lexicon = None
    if arm.lexicon == "sets":
        lexicon = table
    elif arm.lexicon == "random":
        lexicon = np.eye(table.shape[1], dtype=bool)[np.random.default_rng(12345).integers(0, table.shape[1], cfg.vocab)]
    if lexicon is not None:
        params["lex_w"] = rng.normal(0, 0.02, (cfg.width, lexicon.shape[1])).astype(np.float32)
        params["lex_b"] = np.zeros(lexicon.shape[1], dtype=np.float32)
    optimizer = tinylm.AdamW(params)
    senses, roles = corrupt(data, arm, rng) if arm.sense_noise or arm.role_noise else (data.senses, data.roles)
    n_roles, n_senses = cfg.tag_sizes[0] if arm.roles else 0, data.sense_sets.shape[1]
    heads = []  # functions from token positions to the allowed-label mask of each tag head
    if arm.roles:
        heads.append(lambda idx: np.eye(n_roles, dtype=bool)[roles[idx]])
    if arm.senses == "gold":
        heads.append(lambda idx: np.eye(n_senses, dtype=bool)[senses[idx]])
    elif arm.senses == "set":
        heads.append(lambda idx: data.sense_sets[idx])
    elif arm.senses == "random":
        classes = np.random.default_rng(12345).integers(0, n_senses, cfg.vocab)
        heads.append(lambda idx: np.eye(n_senses, dtype=bool)[classes[data.ids[idx]]])
    t = cfg.context
    # Check held-out loss about every half pass over the training text: small sets overfit within a few passes.
    every = max(10, min(settings.eval_every, len(data.ids) // (2 * settings.batch * t)))
    best = (math.inf, params, 0)
    stale = 0
    step = 0
    curve = []
    for step in range(1, settings.max_steps + 1):
        idx = rng.integers(0, len(data.ids) - t, size=settings.batch)[:, None] + np.arange(t + 1)
        x = data.ids[idx]
        logits, tag_logits, cache_ = tinylm.forward(params, cfg, x[:, :-1], rng)
        _, dlogits = tinylm.nll(logits, x[:, 1:])
        dtags = [tinylm.set_nll(tl, head(idx[:, :-1]))[1] for tl, head in zip(tag_logits, heads)]
        grads = tinylm.backward(params, cfg, cache_, dlogits, dtags)
        if lexicon is not None:
            _, lex = tinylm.lexicon_loss(params, lexicon)
            grads["tok"] += lex.pop("tok")
            grads |= lex
        tinylm.clip(grads, 1.0)
        optimizer.step(params, grads, settings.lr * min(1.0, step / settings.warmup))
        if step % every == 0:
            loss = float(token_losses(params, cfg, dev).mean())
            curve.append((step, round(loss, 5)))
            if loss < best[0] - 1e-4:
                best, stale = (loss, {k: v.copy() for k, v in params.items()}, step), 0
            else:
                stale += 1
                if stale >= settings.patience:
                    break
    return best[1], {"dev_loss": best[0], "best_step": best[2], "steps": step, "curve": curve}


def token_losses(params: tinylm.Params, cfg: tinylm.Config, ids: np.ndarray, batch: int = 64) -> np.ndarray:
    """Loss on every token of a held-out text after the first, read in windows of the model's context."""
    t = cfg.context
    starts = np.arange(0, len(ids) - 1, t)
    losses = []
    for chunk in range(0, len(starts), batch):
        s = starts[chunk : chunk + batch]
        idx = np.minimum(s[:, None] + np.arange(t + 1), len(ids) - 1)
        lp = tinylm.logprobs(params, cfg, ids[idx])
        valid = (s[:, None] + np.arange(1, t + 1)) < len(ids)  # the last window runs past the end
        losses.append(-lp[valid])
    return np.concatenate(losses)


def sentence_logprobs(params: tinylm.Params, cfg: tinylm.Config, world: World, sentences: list[list[str]]) -> np.ndarray:
    """Total log-probability of each sentence, read after a sentence end as in running text."""
    word = {w: i for i, w in enumerate(world.vocab)}
    end = word["."]
    length = max(len(s) for s in sentences) + 1
    x = np.full((len(sentences), length), end, dtype=np.int32)
    mask = np.zeros((len(sentences), length - 1), dtype=bool)
    for i, s in enumerate(sentences):
        x[i, 1 : len(s) + 1] = [word[w] for w in s]
        mask[i, : len(s)] = True
    totals = []
    for chunk in range(0, len(sentences), 256):
        totals.append((tinylm.logprobs(params, cfg, x[chunk : chunk + 256]) * mask[chunk : chunk + 256]).sum(1))
    return np.concatenate(totals)


def evaluate(params: tinylm.Params, cfg: tinylm.Config, m: dict, counts: np.ndarray) -> dict:
    world = m["world"]
    losses = token_losses(params, cfg, m["test"])
    seen = counts[m["test"][1:]]
    buckets = {}
    for low, high in BUCKETS:
        chosen = (seen >= low) & (seen <= (high if high is not None else np.inf))
        buckets[f"{low}-{'' if high is None else high}"] = [float(losses[chosen].mean()) if chosen.any() else None, int(chosen.sum())]
    right, wrong = zip(*m["agreement"])
    agreement = sentence_logprobs(params, cfg, world, list(right)) > sentence_logprobs(params, cfg, world, list(wrong))
    word = {w: i for i, w in enumerate(world.vocab)}

    def crossed(items: list[dict]) -> np.ndarray:
        # Sentences 0 and 2 are valid; each word appears once in a valid and once in an invalid one.
        scores = sentence_logprobs(params, cfg, world, [s for item in items for s in item["sentences"]]).reshape(-1, 4)
        return scores[:, 0] - scores[:, 1] + scores[:, 2] - scores[:, 3] > 0

    selection = crossed(m["selection"])
    fewest = np.array([min(counts[word[n]] for n in item["nouns"]) for item in m["selection"]])
    rare_nouns, unseen_nouns = fewest <= RARE, fewest == 0
    sense = crossed(m["sense"])
    return {
        "test_loss": float(losses.mean()),
        "rare_loss": float(losses[seen <= RARE].mean()),
        "buckets": buckets,
        "agreement": float(agreement.mean()),
        "selection": float(selection.mean()),
        "selection_rare": float(selection[rare_nouns].mean()) if rare_nouns.any() else None,
        "selection_rare_n": int(rare_nouns.sum()),
        "selection_unseen": float(selection[unseen_nouns].mean()) if unseen_nouns.any() else None,
        "selection_unseen_n": int(unseen_nouns.sum()),
        "sense": float(sense.mean()),
    }


def run_one(job: tuple[Settings, str, int, int, int]) -> dict:
    settings, arm_name, size, seed, max_size = job
    m = materials(settings, max_size)
    arm = ARMS[arm_name]
    data = Encoded(*(getattr(m["train"], f)[:size] for f in ("ids", "senses", "sense_sets", "roles")))
    cfg = config(settings, m["world"], arm)
    start = time.time()
    params, info = train(settings, cfg, data, m["dev"], arm, seed, lexicon_table(m["world"]))
    counts = np.bincount(data.ids, minlength=len(m["world"].vocab))
    record = {"arm": arm_name, "size": size, "seed": seed, **info, **evaluate(params, cfg, m, counts)}
    record["seconds"] = round(time.time() - start, 1)
    record["settings"] = asdict(settings)
    return record


def run(args: argparse.Namespace) -> None:
    settings = Settings(**json.loads(args.settings)) if args.settings else Settings()
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["settings"] == asdict(settings):
                done.add((r["arm"], r["size"], r["seed"]))
    max_size = max(args.sizes)
    # Largest runs first, so the slowest jobs don't start last.
    jobs = [
        (settings, arm, size, seed, max_size)
        for size in sorted(args.sizes, reverse=True)
        for seed in args.seeds
        for arm in args.arms
        if (arm, size, seed) not in done
    ]
    print(f"{len(jobs)} runs to do, {len(done)} already in {out}", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Each worker uses one thread: small matrices gain little from more, and workers would compete.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool, out.open("a", encoding="utf-8") as f:
        for record in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"{record['arm']:<20} {record['size']:>8,} seed {record['seed']}: test loss {record['test_loss']:.4f}, "
                f"sense {record['sense']:.0%}, {record['seconds']:.0f}s",
                file=sys.stderr,
            )


def report(paths: list[Path]) -> str:
    records = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()]
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        cells[(r["arm"], r["size"])].append(r)
    arms = [a for a in ARMS if any(k[0] == a for k in cells)]
    sizes = sorted({k[1] for k in cells})

    def mean(arm: str, size: int, key: str) -> float | None:
        values = [r[key] for r in cells.get((arm, size), []) if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    def spread(arm: str, size: int, key: str) -> float | None:
        values = [r[key] for r in cells.get((arm, size), []) if r.get(key) is not None]
        if len(values) < 2:
            return None
        mu = sum(values) / len(values)
        return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))

    def table(title: str, key: str, fmt: str) -> list[str]:
        percent = fmt.endswith("%")
        lines = [f"{title} (mean ± standard deviation across seeds)", f"{'arm':<26}" + "".join(f"{s:>16,}" for s in sizes)]
        for arm in arms:
            cells_ = []
            for s in sizes:
                v, sd = mean(arm, s, key), spread(arm, s, key)
                if v is None:
                    cells_.append("-")
                elif sd is None:
                    cells_.append(format(v, fmt))
                else:
                    cells_.append(f"{v:.1%}±{sd * 100:.1f}" if percent else f"{format(v, fmt)}±{format(sd, fmt)}")
            lines.append(f"{arm:<26}" + "".join(f"{c:>16}" for c in cells_))
        return lines + [""]

    lines = [f"{len(records)} runs. Columns are training tokens; cells average up to {max(len(v) for v in cells.values())} seeds.", ""]
    lines += table("Held-out loss on raw text (nats per token, lower is better)", "test_loss", ".4f")
    lines += table(f"Loss on words seen at most {RARE} times in training", "rare_loss", ".3f")
    lines += table("Agreement across a relative clause (accuracy)", "agreement", ".1%")
    lines += table("Meaning constraints on objects (accuracy, chance 50%)", "selection", ".1%")
    lines += table(f"... where a noun was seen at most {RARE} times", "selection_rare", ".1%")
    lines += table("... where a noun never appeared in training", "selection_unseen", ".1%")
    lines += table("Picking a homonym's sense from context (accuracy, chance 50%)", "sense", ".1%")
    for key, title, sign in (
        ("test_loss", "held-out loss", 1),
        ("selection", "meaning-constraint accuracy", -1),
        ("agreement", "agreement accuracy", -1),
        ("sense", "sense-in-context accuracy", -1),
    ):
        plain = [(s, sign * v) for s in sizes if (v := mean("plain", s, key)) is not None]
        if len(plain) < 2:
            continue
        lines.append(f"Text multiplier on {title}: how many times more text the plain model needs to match each arm")
        lines.append(f"{'arm':<26}" + "".join(f"{s:>16,}" for s in sizes))
        for arm in arms:
            row = [_multiplier(plain, s, None if (v := mean(arm, s, key)) is None else sign * v) for s in sizes]
            lines.append(f"{arm:<26}" + "".join(f"{v:>16}" for v in row))
        lines.append("")
    lines.append("Multipliers interpolate the plain model's results linearly in log text size; '>' means past the largest size.")
    lines.append("")
    lines.append("Training steps to first reach the plain model's mean best held-out loss (mean over seeds that reach it)")
    lines.append(f"{'arm':<26}" + "".join(f"{s:>16,}" for s in sizes))
    for arm in arms:
        row = []
        for s in sizes:
            target = mean("plain", s, "dev_loss")
            runs = [r for r in cells.get((arm, s), []) if "curve" in r]
            if target is None or not runs:
                row.append("-")
                continue
            hits = [next((step for step, loss in r["curve"] if loss <= target + 1e-4), None) for r in runs]
            reached = [h for h in hits if h is not None]
            row.append(f"{sum(reached) / len(reached):.0f} ({len(reached)}/{len(runs)})" if reached else f"never (0/{len(runs)})")
        lines.append(f"{arm:<26}" + "".join(f"{c:>16}" for c in row))
    return "\n".join(lines)


def _multiplier(plain: list[tuple[int, float]], size: int, loss: float | None) -> str:
    if loss is None:
        return "-"
    for (s0, l0), (s1, l1) in zip(plain, plain[1:]):
        if l1 <= loss <= l0 and l0 != l1:
            log_size = math.log(s0) + (l0 - loss) / (l0 - l1) * (math.log(s1) - math.log(s0))
            return f"{math.exp(log_size) / size:.2f}x"
    if loss < plain[-1][1]:
        return f">{plain[-1][0] / size:.1f}x"
    return f"<{plain[0][0] / size:.2f}x"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tagexp", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="train every arm at every size and seed, appending results as JSON lines")
    r.add_argument("--out", default="results/tagexp.jsonl")
    r.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    r.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--settings", help="JSON overriding fields of Settings, e.g. '{\"dropout\": 0.2}'")
    p = sub.add_parser("report", help="summarise one or more results files")
    p.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        print(report(args.paths))


if __name__ == "__main__":
    main()
