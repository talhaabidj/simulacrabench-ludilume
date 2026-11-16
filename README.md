# SimulacraBench entry: Ludilume Studios

[![checks](https://github.com/talhaabidj/simulacrabench-ludilume/actions/workflows/checks.yml/badge.svg)](https://github.com/talhaabidj/simulacrabench-ludilume/actions/workflows/checks.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Code for the Test submission of team Ludilume Studios (Talha Abid, University of
Central Punjab, Pakistan) to [SimulacraBench](https://www.codabench.org/competitions/17822/),
a NeurIPS 2026 competition. The task is to predict, for each held-out survey
answer, a probability for every answer option, on three instruments from UNICEF,
the World Bank and UNHCR.

This repository holds the exact files that were scored, and two scripts that
rebuild the submitted zip byte for byte.

## Contents

- [At a glance](#at-a-glance)
- [Quick start](#quick-start)
- [What is in this repository](#what-is-in-this-repository)
- [How the method works](#how-the-method-works)
- [How the submission runs](#how-the-submission-runs)
- [Rebuild and verify](#rebuild-and-verify)
- [Run it on practice data](#run-it-on-practice-data)
- [Hardware and time](#hardware-and-time)
- [Pretrained models](#pretrained-models)
- [Third-party code in the zip](#third-party-code-in-the-zip)
- [Data](#data)
- [Evidence in short](#evidence-in-short)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)

## At a glance

| | |
|---|---|
| Competition | SimulacraBench, NeurIPS 2026 competition track ([simulacrabench.org](https://simulacrabench.org/)) |
| Team | Ludilume Studios (Codabench participant `Talha`) |
| Scored submission | Test phase, Codabench submission 1053745, 15 November 2026 |
| Submitted file | `submission.zip`, 943,721,566 bytes, SHA-256 `ec48ff5629f21732c38bf83147571bf1bb049211c03c3286464f35843b0f9c5b` |
| What this repository reproduces | that file, byte for byte |
| What it cannot reproduce | the score: the survey data are not public |
| Licence of our code | MIT |

Skill is 0 for the crowd marginal and 1 for a perfect prediction.

| | Grand Prize | UNICEF | World Bank | UNHCR |
|---|---|---|---|---|
| Test (1053745) | 0.4862 | 0.3426 | 0.2368 | 0.8793 |
| Development, the same zip (1052251) | 0.4878 | 0.3499 | 0.2486 | 0.8650 |
| Development, our leaderboard entry (1039127, an earlier build) | 0.4943 | 0.3499 | 0.2486 | 0.8843 |

The leaderboard shows these rounded to two decimals. Development readings are
noised and rounded by the platform and are not comparable with Test.

## Quick start

You need Python 3.10 or later, git, about 2 GB of free disk space and an
internet connection. No GPU and no extra packages are needed. If your system
calls Python 3 `python`, use that in place of `python3`.

```bash
git clone https://github.com/talhaabidj/simulacrabench-ludilume
cd simulacrabench-ludilume
git checkout test-submission-1053745
python3 scripts/fetch_weights.py
python3 scripts/build_submission.py
```

The last lines should read:

```
156 files match manifest.json
.../dist/submission.zip
SHA-256   ec48ff5629f21732c38bf83147571bf1bb049211c03c3286464f35843b0f9c5b
submitted ec48ff5629f21732c38bf83147571bf1bb049211c03c3286464f35843b0f9c5b
The zip is byte-identical to the Test submission.
```

The tag `test-submission-1053745` marks the commit that the technical report
cites. The files under `submission/` and `manifest.json` are the ones that were
scored; they have not changed since they were added to this repository.

## What is in this repository

| Path | What it is |
|---|---|
| `submission/main.py` | The submission: defines `predict(frame, schema)`. Unchanged since it was scored |
| `submission/requirements.txt` | Packages the hosted image lacks, pinned to exact versions |
| `submission/libgomp-*.so.1`, `libgomp-copyright.txt` | The OpenMP runtime that LightGBM needs, loaded only when the system has none |
| `submission/sdm/`, `submission/licenses/`, `submission/NOTICE` | Third-party code and licences that were in the zip (see [below](#third-party-code-in-the-zip)) |
| `manifest.json` | SHA-256 and size of every file in the submitted zip, the zip's own hash, and the pinned source of each model file |
| `scripts/fetch_weights.py` | Downloads the three model files at their pinned commits and checks their hashes |
| `scripts/build_submission.py` | Checks every file against the manifest and writes `dist/submission.zip` |
| `scripts/manifest.py` | Helpers the two scripts share |
| `scripts/test_*.py` | Checks of the repository that need no GPU and no model files |
| `requirements-dev.txt` | The pinned tool that runs the checks |
| `.github/workflows/checks.yml` | Runs the checks on Linux, macOS and Windows for every change |
| `CITATION.cff` | How to cite this repository |

The three model files (about 1 GB) are not in the repository. They are fetched
from their authors' pages and checked by hash.

A few comments in `main.py` mention `REVIEW.md`. That is a private development
log and is not part of this repository.
