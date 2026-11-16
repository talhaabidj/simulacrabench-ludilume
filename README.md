# SimulacraBench entry: Ludilume Studios

Code for the Test submission of team Ludilume Studios (Talha Abid, University of
Central Punjab, Pakistan) to [SimulacraBench](https://www.codabench.org/competitions/17822/),
a NeurIPS 2026 competition. The task is to predict, for each held-out survey
answer, a probability for every answer option, on three instruments from UNICEF,
the World Bank and UNHCR.

This repository holds the exact files that were scored, and two scripts that
rebuild the submitted zip byte for byte.

## Result

Test phase, Codabench submission 1053745 (15 November 2026). Skill is 0 for the
crowd marginal and 1 for a perfect prediction.

| | Grand | UNICEF | World Bank | UNHCR |
|---|---|---|---|---|
| Test | 0.4862 | 0.3426 | 0.2368 | 0.8793 |
| Development (noised and rounded by the platform) | 0.4878 | 0.3499 | 0.2486 | 0.8650 |

## Method in short

Everything is fitted inside `predict()` on the respondents whose answers are
visible. No model is trained beforehand on survey data.

1. **Candidates per question.** For each scored question, several models
   predict the answer from the respondent's always-visible background answers:
   - the crowd marginal, and count tables over the most informative background
     columns, smoothed toward coarser tables;
   - regularised multinomial logistic regression, with and without the most
     informative column pairs;
   - gradient-boosted trees (LightGBM) in several settings, including one with
     target-encoded inputs and one for questions with ordered options;
   - multi-task neural networks that share one trunk across all questions;
   - three tabular foundation models used in context, without fine-tuning:
     TabICL v2, TabPFN v2 and TabPFN-3.5.
2. **Skip logic.** When a question's gate parent is a background answer and the
   visible rows obey the rule, closed rows get the "not asked" option directly.
   When the parent is itself a scored question, extra candidates combine the
   parent's predicted answer with a model fitted on the rows that were asked.
3. **Household matching (UNHCR).** The UNHCR file holds two survey rounds, and
   its schema gives no identifier that links a household across them. In case
   some households were interviewed in both, rows from the two rounds are
   matched when each is the other's best match by a clear margin, first on
   their answers and then, at a lower bar, where the background columns also
   agree. A matched partner's answers give two more candidates per question. The
   step stays off when it finds fewer than 30 matches, and the hosted run does
   not tell us whether it switched on. We asked the organizers whether a
   model may benefit from such matches, using only the GIVEN columns and the
   visible rows provided at run time; they replied by email (6 November 2026)
   that this is permitted. Nothing is matched to any outside source, and no
   person is identified.
4. **Blend.** Every candidate predicts the visible rows out of fold. Per
   question, mixture weights are fitted by EM to maximise the grader's own
   floored log score on those predictions, and the better of a linear, a
   tempered or a log pool is kept. The held-out rows get that blend.
5. **Clock.** The three instruments share one time limit. Work runs in passes,
   cheap candidates first, so a slow machine loses accuracy, not answers. Any
   candidate that fails is dropped; a question with nothing left gets the crowd
   marginal. On a real instrument the run stops with an error if TabICL or
   TabPFN-3.5 cannot load, because a failed Test run may be repeated and a weak
   finished one may not.

The technical report on OpenReview describes the method and its evidence in
full.

## Layout

| Path | What it is |
|---|---|
| `submission/main.py` | The submission: defines `predict(frame, schema)`. Unchanged since it was scored |
| `submission/requirements.txt` | Packages the hosted image lacks, pinned |
| `submission/libgomp-*.so.1`, `libgomp-copyright.txt` | Debian's OpenMP runtime for LightGBM, loaded only when the system has none |
| `submission/sdm/`, `submission/licenses/`, `submission/NOTICE` | Third-party code and licences that were in the zip (see below) |
| `manifest.json` | SHA-256 of every file in the submitted zip, the zip's own hash, and the pinned model files |
| `scripts/fetch_weights.py` | Downloads the three model files at their pinned commits and checks their hashes |
| `scripts/build_submission.py` | Checks every file against the manifest and writes `dist/submission.zip` |
| `scripts/test_*.py` | Checks of the repository that need no GPU and no model files |

A few comments in `main.py` mention `REVIEW.md`. That is a private development
log and is not part of this repository.

## Reproduce the submission file

You need Python 3.10 or later, about 2 GB of free disk space and an internet
connection. The scripts use only the standard library. No GPU is needed. If
your system calls Python 3 `python`, use that in place of `python3`.

```bash
git clone https://github.com/talhaabidj/simulacrabench-ludilume
cd simulacrabench-ludilume
git checkout test-submission-1053745
python3 scripts/fetch_weights.py
python3 scripts/build_submission.py
```

`fetch_weights.py` downloads three files, about 1 GB together, from Hugging
Face. No account is needed. Each file is kept only if its size and SHA-256
match `manifest.json`. `build_submission.py` then checks all 156 files and
writes `dist/submission.zip`. The whole thing takes a few minutes; most of it
is the download.

The tag `test-submission-1053745` marks the commit that the technical report
cites. The files under `submission/` and `manifest.json` are the ones that were
scored; they have not changed since the first commit of this repository.

## Verify

The last lines the build prints are the SHA-256 of the zip it wrote and that of
the zip that was scored:

```
ec48ff5629f21732c38bf83147571bf1bb049211c03c3286464f35843b0f9c5b
```

The zip is written with fixed names, order, dates and permissions. With zlib
1.2.12 it is byte-identical to the submitted file (943,721,566 bytes). With
another zlib version the compressed bytes can differ; the script then says so,
and the 156 files inside are still identical, each checked by hash.

To check the zip against the competition's contract, use the official starter
kit (<https://github.com/SituatedEvals/public>, commit `aa03186`):

```bash
python3 tools/check_submission_zip.py /path/to/dist/submission.zip
```

To run the repository's own checks:

```bash
python3 -m pip install pytest
python3 -m pytest scripts
```

## Hardware and time

The real survey data are not public, so real scores come only from the hosted
grader. There the zip ran on Python 3.13 with one H100 GPU, 8 CPUs and 16 GB of
memory, under a limit of 3,600 seconds for the three instruments together. Our
Test run took about 36 minutes.

To run `predict()` yourself, use the starter kit's `score.py` with its synthetic
sandbox data. It installs `submission/requirements.txt` into a fresh
environment. Our own full-length rehearsals ran on a 16 GB T4 GPU and used
nearly the whole 3,600 seconds. We did not test a run without a GPU.

## Limitations

- The score cannot be reproduced outside the hosted grader, because the survey
  data are not released. What this repository reproduces exactly is the
  submitted file.
- The run is driven by the clock: it does as much as fits in the time limit.
  Seeds are fixed, but a different machine finishes a different amount of work,
  so two runs can give slightly different predictions.
- The three model files are hosted by their authors. If one is moved or removed,
  `fetch_weights.py` will fail for that file; the hashes in `manifest.json`
  still identify the exact files.
- `submission/main.py` is kept byte for byte as it was scored. It is one long
  file and carries comments from development.
