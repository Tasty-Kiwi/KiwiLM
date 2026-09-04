# Extended five-seed generation analysis

## Verdict

The larger sample confirms that **500M tokens did not fix phrase and sentence
loops**, especially with low-temperature sampling. However, it also finds a
real improvement hidden by the original seed-42 suite: the 500M checkpoint has
**zero single-word collapses**, versus six for AdamW 250M and two for Muon 250M.

This is a mixed behavioral result. Scaling removed the most extreme failure
mode, but did not reduce the broader tendency to repeat phrases.

## Test design

- Three exact checkpoints: Dense AdamW 250M, Dense Muon 0.01 250M, and Dense
  Muon 0.01 500M.
- Twelve prompts: six existing story-continuity prompts and six new
  expository continuations covering science, history, computing, a procedure,
  and a general explanation.
- Focused sampling: temperature 0.4, top-k 20. Creative sampling: temperature
  0.8, top-k 40.
- Seeds 42–46, maximum 160 new tokens, FP32 MPS, cache disabled.
- 120 samples per checkpoint, 360 total. Each comparison position has the same
  prompt, profile, and seed. The evaluator verifies checkpoint hashes against
  the primary comparison before inference.

These prompts do not establish general factual accuracy. They expose
continuation behavior in base pretrained models; they are not chat or
instruction-following tests.

## Aggregate result

| Model | Mean repeated 4-gram rate | Samples above 0.5 | Word collapses of 20+ | Maximum word run |
| --- | ---: | ---: | ---: | ---: |
| Dense AdamW 250M | **0.2896** | 36/120 | 6/120 | 158 |
| Dense Muon 250M | 0.3014 | **34/120** | 2/120 | 130 |
| Dense Muon 500M | 0.3097 | 39/120 | **0/120** | **2** |

On matched samples, 500M has less four-gram repetition than Muon 250M in 51
positions, more in 64, and ties in five. The mean paired increase is 0.0083 and
the median increase is 0.0075. Five seeds are useful evidence against a
seed-42 accident, but they are not enough to attach a narrow confidence claim.
The per-seed aggregate means are not monotonic: 500M is worse on seeds 42, 44,
and 45 and better on 43 and 46 compared with Muon 250M.

## Story prompts

| Model | Focused mean | Focused >0.5 | Creative mean | Creative >0.5 |
| --- | ---: | ---: | ---: | ---: |
| Dense AdamW 250M | **0.4067** | **13/30** | **0.0312** | **0/30** |
| Dense Muon 250M | 0.4634 | **13/30** | 0.0785 | 1/30 |
| Dense Muon 500M | 0.4832 | 15/30 | 0.0645 | 1/30 |

Scaling slightly improves creative repetition relative to Muon 250M, but
worsens focused repetition and severe-loop incidence. The earlier 12-sample
warning was therefore directionally representative, not just an unlucky seed.

## Expository prompts

| Model | Focused mean | Focused >0.5 | Creative mean | Creative >0.5 |
| --- | ---: | ---: | ---: | ---: |
| Dense AdamW 250M | **0.6113** | **21/30** | 0.1094 | 2/30 |
| Dense Muon 250M | 0.6001 | 20/30 | 0.0636 | **0/30** |
| Dense Muon 500M | 0.6470 | 23/30 | **0.0441** | **0/30** |

The sampling regime matters more than the checkpoint difference. Focused
sampling loops badly for every model. Creative sampling sharply reduces
repetition, and 500M is best by this narrow metric, but manual inspection shows
that many creative continuations are incoherent or factually wrong. For
example, water-cycle continuations confuse evaporation and condensation, and
tea instructions introduce bowls, vinegar, salt, or unrelated cooking steps.
Low repetition is therefore not equivalent to high quality.

## Failure-mode change

AdamW collapses to the word `print` for 158 tokens in two focused history
samples and 135 tokens in one creative sample. It also has a 97-token `Dr`
collapse. Muon 250M has a 130-token `Great` collapse and a 51-token `printing`
collapse. Muon 500M never repeats one word more than twice in any of its 120
samples.

The 500M failures instead repeat whole clauses or list items. Examples include
`The map is a map ...`, repeated tea-preparation steps, and repeated statements
about water entering the atmosphere. Four-gram repetition correctly detects
these even when the consecutive-word metric does not.

## Implication

Do not expect a 1B run alone to cure degeneration. If continuing to 1B, retain
this exact suite as a predeclared regression test and evaluate alternative
decoding separately from model training. A temperature/top-k sweep can test
whether the focused regime is simply too sharp, but it cannot substitute for
model-quality evaluation: the creative samples demonstrate the repetition
versus coherence tradeoff.

Artifacts:

- [Machine summary](summary.json)
- [Raw rows](results.jsonl)
- [All 360 samples](samples.md)
- [Exact suite](suite.json)

The suite deliberately stores every generated sample; no examples were removed
after inspection.

The subsequent [decoding sweep](../decoding-sweep/analysis.md) finds 0.8/40 to
be the conservative default and 0.8/80 to be a useful anti-repetition profile.
