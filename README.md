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
