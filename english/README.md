# The label experiment on English

The synthetic experiment (`src/hevdicgrm/tagexp.py`) tests whether sense and role labels let a model learn from less
text, in a language where the labels are exact. This directory runs the same comparison on real English, where they
aren't exact:

- **roles** come from spaCy's parser;
- **senses** come from WordNet's 45 supersenses, such as `noun.food` or `verb.consumption`.

None of the arms below needs Jev, so the experiment costs only GPU time.

Not yet run: this repository's environment has no PyTorch, spaCy, NLTK or data. `labels.py`, which moves word labels
onto subword tokens, is tested in `tests/test_english_labels.py`. Run the smoke test before renting a GPU.

## Arms

| Arm | Extra targets for each word's first subword |
| --- | --- |
| `plain` | none |
| `roles` | the word's dependency relation (spaCy `en_core_web_sm`) |
| `sense-sets` | every supersense WordNet gives the word's lemma for its part of speech |
| `roles+sense-sets` | both |
| `lexicon` | none in the text; each whole-word token's embedding must predict its WordNet supersenses |
| `roles+sense-sets+lexicon` | all of the above |
| `random-classes` | a control: a fixed random class per token, which regularizes but carries no meaning |

## Steps

1. Get the BabyLM strict-small corpus (10M words) from [babylm.github.io](https://babylm.github.io/) and the BLiMP
   jsonl files from [github.com/alexwarstadt/blimp](https://github.com/alexwarstadt/blimp/tree/master/data).
2. Tag the corpus. spaCy takes about 15 minutes for 10M words with 4 processes.

   ```sh
   pip install spacy nltk tokenizers numpy "torch>=2.1"
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('wordnet')"
   python english/prepare.py --train babylm/train_10M/*.train --dev babylm/dev/*.dev --out data/babylm
   ```

3. Check the pipeline on CPU in about a minute:

   ```sh
   python english/train.py --data data/babylm --arm roles+sense-sets+lexicon --smoke --blimp blimp/data --out results/smoke.jsonl
   ```

4. Run the grid: 7 arms × 3 sizes × 2 seeds.

   ```sh
   for tokens in 1000000 3000000 13000000; do for seed in 0 1; do
     for arm in plain roles sense-sets roles+sense-sets lexicon roles+sense-sets+lexicon random-classes; do
       python english/train.py --data data/babylm --arm $arm --train-tokens $tokens --seed $seed --blimp blimp/data
   done; done; done
   ```

   The default model has 6 layers and width 384, about 17M parameters. Each run trains until held-out loss stops
   improving, for at most 20,000 steps of 8,192 tokens. That should take minutes per run on one modern GPU, and a
   few GPU-hours for the whole grid, but this hasn't been measured.

## Reading the result

Each run appends to `results/english.jsonl`:
- the held-out loss;
- accuracy on each of BLiMP's 67 paradigms, each tagged with its phenomenon;
- the mean over paradigms.

The synthetic experiment predicts two things:
- the gains appear at the smallest size and shrink as text grows;
- roles help agreement (`subject-verb agreement`, `determiner-noun agreement`), and sense sets help `argument structure`.

| What you see | What it means | Next step |
| --- | --- | --- |
| A labelled arm beats `plain` at 1M tokens by more than the seed spread, and `random-classes` doesn't | The mechanism transfers to English with free labels | Try a low-resource language |
| Gains at 1M tokens that vanish by 13M | The same as the synthetic result: a tool for scarce text | Aim it at low-resource languages and rare words |
| `random-classes` does as well as the labelled arms | Extra heads regularize; the labels' content isn't what helps | Stop, or try richer labels |
| No arm beats `plain` | WordNet and the parser don't carry what the model needs | Stop before paying for Jev labels |

For a low-resource language, the parser and WordNet don't exist. The grammar book's interlinear glossed examples
already give each word its meaning and grammatical role, and [GlossLM](https://aclanthology.org/2024.emnlp-main.683)
collects about 450,000 such examples in about 1,800 languages.
