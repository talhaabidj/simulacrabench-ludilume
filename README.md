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

| | Grand Prize | UNICEF | World Bank | UNHCR |
|---|---|---|---|---|
| Test (1053745) | 0.4862 | 0.3426 | 0.2368 | 0.8793 |
| Development, the same zip (1052251) | 0.4878 | 0.3499 | 0.2486 | 0.8650 |
| Development, our leaderboard entry (1039127, an earlier build) | 0.4943 | 0.3499 | 0.2486 | 0.8843 |

The leaderboard shows these rounded to two decimals. Development readings are
noised and rounded by the platform and are not comparable with Test.

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
3. **Household matching (UNHCR).** The UNHCR file holds two survey rounds.
   UNHCR's public note on the survey's second wave says that it recontacted the
   first-wave households (see Data below). The competition schema neither says
   that its rounds are those waves nor gives an identifier that links a
   household across them. In case households do appear in both rounds, rows
   from the two rounds are matched when each is the other's best match by a
   clear margin, first on
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
match `manifest.json`. If the connection drops, the script continues from where
it stopped; if it gives up, run it again. `build_submission.py` then checks all
156 files and writes `dist/submission.zip`. The download takes from a few
minutes to half an hour, depending on the connection; the build takes under a
minute.

The tag `test-submission-1053745` marks the commit that the technical report
cites. The files under `submission/` and `manifest.json` are the ones that were
scored; they have not changed since they were added to this repository.

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
memory, under a limit of 3,600 seconds for the three instruments together. The
hosted image already holds PyTorch, scikit-learn and SciPy; the other packages
are pinned in `submission/requirements.txt`. Our Test run took about 36
minutes.

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
- The hosted logs are hidden, so we cannot confirm that household matching
  switched on in the real UNHCR file. Its evidence is two panels built from one
  other public survey, with skip patterns of our own making; rows without a
  matched partner lost a little there.
- Later builds were chosen on public stand-in surveys, so their gains on the
  real data are unmeasured.
- Hidden answers are predicted one by one. There is no joint model of a
  respondent's answers.

## Pretrained models

All three are public, unmodified, and bundled in the zip.

| Model | Hugging Face repository and commit | File | SHA-256 | Licence |
|---|---|---|---|---|
| TabICL v2 | `jingang/TabICL` @ `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | `tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` | BSD-3-Clause |
| TabPFN v2 | `Prior-Labs/TabPFN-v2-clf` @ `275f19d2add9c2c0de38f9ebaded3856599fa2e7` | `tabpfn-v2-classifier-v2_default.ckpt` | `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49` | Prior Labs License 1.1 |
| TabPFN-3.5 | `Prior-Labs/tabpfn_3_5` @ `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e` | `tabpfn-v3.5-20260909_multiclass.safetensors` | `4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149` | TabPFN-3.5 Non-Commercial License |

Built with PriorLabs-TabPFN.

## Third-party code in the zip

- `submission/sdm/` is structured-data-models 0.2.1 (Apache-2.0). It was bundled
  for a model that the final build does not use; no prediction depends on it. It
  is kept here so that the rebuilt zip matches the submitted one.
- `submission/libgomp-*.so.1` are the GNU OpenMP runtime from Debian 11
  (bullseye), package `libgomp1`, for x86_64 and aarch64. They are under GPL-3
  with the GCC Runtime Library Exception; see
  `submission/libgomp-copyright.txt`. The source is Debian's `gcc-10` package
  (<https://sources.debian.org/src/gcc-10/>).
- Licences for the above and for TabPFN-3.5 are in `submission/licenses/` and
  `submission/NOTICE`.

## Data

This repository holds no survey data. The submission uses only the frame the
grader passes to `predict()`.

Public survey data were used offline, on Kaggle and Colab, to compare builds
before uploading. They never entered the submission and were never linked to
competition data:

| Data | Obtained from | Licence |
|---|---|---|
| Afrobarometer, rounds 6 to 9 | copies published by Electric Sheep Africa on Hugging Face (`electricsheepafrica/africa-afrobarometer-<country>`); original: <https://www.afrobarometer.org/data/> | CC-BY-4.0 (the copies) |
| Encuesta Permanente de Hogares, Argentina. Source: Instituto Nacional de Estadística y Censos (INDEC) | `Matiasdn/eph-microdata-parquet` on Hugging Face | public microdata; the copy carries no licence tag |
| Rwanda Labour Force Survey 2019, 2022 and 2024, National Institute of Statistics of Rwanda | copies published by Electric Sheep Africa on Hugging Face; original: <https://microdata.statistics.gov.rw/> | CC-BY-4.0 (the copies) |
| RIASEC test answers, Open-Source Psychometrics Project | <https://openpsychometrics.org/_rawdata/> | none stated |

For context we also read the competition's public starter kit and UNHCR's two
public methodology notes on the survey behind the UNHCR instrument
(<https://data.unhcr.org/en/documents/details/118503> and
<https://data.unhcr.org/en/documents/details/122187>). These are documents, not
data.

## Licence

The code in this repository that is ours is under the MIT licence (see
`LICENSE`). The pretrained models and the third-party code keep their own
licences, listed above.
