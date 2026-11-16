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
