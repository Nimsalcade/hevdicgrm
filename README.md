# hevdicgrm

Experiments that combine Jev (TypeSafe AI's judgment model), a dictionary and a grammar.

## normcheck: the Meaning Atlas's first test

The Meaning Atlas plan asks Jev hundreds of rating questions about every dictionary sense. Before paying for that,
`normcheck` measures how closely Jev's ratings track human ratings that already exist:

- **[Glasgow Norms](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538586/)**: 5,553 words rated on nine scales
  (valence, arousal, concreteness, size and others). 379 ambiguous words are also rated one sense at a time,
  written like `toast (bread)`.
- **[Lancaster Sensorimotor Norms](https://www.lancaster.ac.uk/psychology/lsnorms/)**: 39,707 words rated for how
  strongly each is experienced through six senses and acted on with five parts of the body.

Each word gets one request that asks every scale as a Jev Score question with the same levels the human raters saw.
For each scale the report gives:

- **rho**: the Spearman correlation between Jev's expected score and the human mean, with a 95% interval.
- **same order** (Glasgow only): take the pairs of senses of one word that people rated clearly apart (by at least
  10% of the scale). This is the share of those pairs Jev puts in the same order. Chance is 50%.

The second number is the one that decides the Atlas. A model can rank words well and still ignore which sense it
was asked about, and then a per-sense Atlas is a per-word Atlas you paid for many times over.

### Run it

```sh
pip install -e .
export TYPESAFE_API_KEY=...
normcheck glasgow.csv --battery batteries/glasgow.json --limit 0     # all 5,553 words, about $0.13
normcheck lancaster.csv --battery batteries/lancaster.json           # a 1,000-word sample, about $0.02
```

Costs are estimates at $0.042 per million input tokens, assuming about four characters per token. The report
prints the real figure from the token counts Jev returns. Answers are saved to `runs/<battery>.jsonl` as they
arrive, so an interrupted run resumes where it stopped and a rerun costs nothing. Changing the questions or the
model makes `normcheck` ask again.

**Data.** The Lancaster CSV is on [OSF](https://osf.io/7emr6/). The battery's column names (`Word`, `Haptic.mean`,
...) haven't been checked against that file. If they differ, `normcheck` stops and lists the real headers, and the
fix is to edit the battery's `column` fields. For Glasgow, make a CSV with a `word` column plus the nine scale means
under the paper's codes: `AROU`, `VAL`, `DOM`, `CNC`, `IMAG`, `FAM`, `AOA`, `SIZE`, `GEND`.

### Reading the result

The bar for rho is how well one half of the human raters predicts the other half. A Jev rho near that means Jev
rates like another panel of people.

| What you see | What it means | Next step |
| --- | --- | --- |
| High rho, and sense pairs mostly in the human order | Jev rates senses, not just words | Run the Atlas per sense |
| High rho, but sense pairs near 50% | Jev rates the word and ignores the cue | Build the Atlas per word, not per sense |
| Low rho on `hand_arm`, `foot_leg` and the other action scales | The same gap [Xu et al. (2025)](https://www.nature.com/articles/s41562-025-02203-8) found in other text-only models | Check action questions like "can you hold it?" against people before they go into the Atlas |

### Battery format

A battery is a JSON file:

- `word_column`: the CSV column with the words.
- `state_template`: what Jev is told about each word, with `{word}` where the word goes.
- `lowercase` (optional): lowercase each word before asking.
- `dimensions`: one entry per scale. Each entry has the CSV `column` with the human means, the `instructions`
  (the question), and `levels`, one per point on the human scale, lowest first.

The levels must match the human scale point for point, because the sense check measures gaps in scale points. The
shipped batteries paraphrase the original rating instructions rather than quoting them.

## tagexp: can labels stand in for text?

The plan's training bet is that a model learns faster from text whose words carry their sense and their role in
the sentence. Real text can't test that cleanly, because its senses and roles are never known exactly. So `tagexp`
runs the test in a synthetic language (`toyworld`) that is generated from senses and roles. It has the
properties the bet depends on:

- **Homonyms.** 40% of content-word tokens are ambiguous between two senses.
- **Meaning constraints.** "eat" needs food, and "clever" needs a person or an idea.
- **Grammar.** Number agreement holds across relative clauses.
- **A long tail of rare words.** Words within a class follow a Zipf distribution.

Every arm trains the same small transformer (`tinylm`: NumPy, gradients checked against finite differences) on the
same text. The arms differ only in what else the model must predict for each token:

| Arm | Extra targets | Real-world source |
| --- | --- | --- |
| `plain` | none | |
| `roles` | grammatical role | a parser |
| `sense-sets` | every sense the word can have | a dictionary alone, as in SenseBERT |
| `senses` | the sense in context | a dictionary plus a perfect Jev |
| `roles+sense-sets` | both cheap labels | parser and dictionary, no Jev |
| `roles+senses` | everything | |
| `roles+senses-noisy` | 30% of ambiguous words get the wrong sense, 10% of roles are wrong | an imperfect Jev and parser |
| `random-classes` | a fixed random class per word | a control: regularizes like a sense head but means nothing |

Every model is tested on raw text alone, so labels can only help through what the model learned from them. Each run
trains until held-out loss stops improving, then reports:

- held-out loss;
- accuracy on agreement across a relative clause;
- accuracy on meaning constraints (a 2×2 design that cancels out word frequency);
- accuracy on picking a homonym's sense from an adjective that cues it.

```sh
PYTHONPATH=src python -m hevdicgrm.tagexp run --out results/tagexp.jsonl --seeds 0 1 2
PYTHONPATH=src python -m hevdicgrm.tagexp report results/tagexp.jsonl
```

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

The tests use fakes in place of Jev, so they run without the SDK, an API key or network access.
