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
| `coarse-sense-sets` | sense lists over merged classes, so "eat" and "drink" share a label | a dictionary coarser than the language |
| `lexicon` | none in the text; each word's embedding must predict that word's senses, seen in the text or not | a dictionary used as a table |
| `roles+sense-sets+lexicon` | everything that needs no Jev | parser and dictionary |
| `random-classes` | a fixed random class per word | a control: an extra head that means nothing |
| `random-lexicon` | a fixed random class per word, as a lexicon | a control for `lexicon` |

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

### Results

Every arm ran with 3 seeds at 10k, 20k and 40k training tokens. At 80k and 200k, every arm is near its ceiling,
so `plain` ran with 3 seeds and the rest with 1. `report` prints every table with its standard deviations. Each run
takes 1 to 9 minutes on one CPU core.

Accuracy on meaning constraints (chance is 50%):

| Arm | 10k | 20k | 40k | 80k |
| --- | --- | --- | --- | --- |
| `plain` | 46% | 59% | 77% | 99.7% |
| `sense-sets` | 54% | 97% | 99.7% | 100% |
| `senses` | 54% | 98% | 99.8% | 100% |
| `lexicon` | 58% | 98% | 100% | 100% |
| `roles+sense-sets+lexicon` | 97% | 99.8% | 99.9% | 100% |
| `random-classes` | 50% | 58% | 78% | 98.7% |
| `random-lexicon` | 49% | 58% | 73% | 98.7% |

What the runs show:

1. **Dictionary labels make meaning constraints learnable from much less text.** With the three labels that need
   no Jev, the model gets meaning constraints right 97% of the time from 10k tokens. The plain model needs about 74k
   tokens, about 7× as much, to match it. Sense lists alone at 20k tokens are worth about 3.6× the text.
2. **The labels' content does the work.** Random per-word classes, with the same heads and number of labels, leave
   meaning constraints where the plain model has them. At 20k tokens, sense lists beat random classes by
   38.8 ± 1.4 points.
3. **Jev adds little on top of the dictionary.** Predicting each word's sense in context, instead of the list of
   senses it can have, adds 1.4 ± 0.7 points on meaning constraints at 20k tokens and 3 to 6 points on picking a
   homonym's sense.
4. **Wrong labels hurt only when text is scarcest.** With 30% of ambiguous words given the wrong sense and 10% of
   roles wrong, accuracy drops 20 points at 10k tokens and not at all from 20k on. Merging classes into a coarser
   inventory costs 3 to 7 points.
5. **The dictionary covers words the text never shows.** UNSEEN
6. **The grammar labels' effect on agreement is clear only at the smallest size.** At 10k tokens, roles add
   11.6 ± 3.0 points on agreement and random classes add 1.3 ± 2.8. At 40k tokens, random classes add as much as
   roles do (27 against 23 points), so past the smallest size an extra per-word head helps agreement whatever it
   predicts. DROPOUT
7. **Held-out loss improves by about twice the text's worth.** With the three no-Jev labels, the loss on plain text
   is as low as the plain model reaches with 1.8× the text at 10k tokens, 2.3× at 20k, 1.8× at 40k, 1.45× at 80k
   and 1.0× at 200k.
8. **With enough text, the plain model catches up.** By 80k tokens it gets 99.7% on meaning constraints.

BIGGER

**What this means for the plan.** In this language, dictionary and grammar labels are worth several times more text,
but only while text is scarce: the plain model catches up once it has seen each word a few dozen times. That makes
the idea a tool for scarce text (low-resource languages, rare words, small models), not a way around large-scale
training. Most of the value comes from the dictionary, so it doesn't need Jev. This language was generated from
exactly the structure the labels describe, so these gains are an upper bound. Whether they carry over to English,
where the labels come from WordNet and a parser, is what `english/` tests.

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

The tests use fakes in place of Jev, so they run without the SDK, an API key or network access.
