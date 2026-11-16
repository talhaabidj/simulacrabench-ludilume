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
