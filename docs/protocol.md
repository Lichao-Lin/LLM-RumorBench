# Evaluation protocol

## 1. Scope and provenance

The legacy repository establishes four condition names: Neither, web search,
deep think, Both. Their interpretation as a search × thinking 2×2 design is an
extension convention, not a recovered original experimental protocol.
Confirm the actual tool settings before analyzing new data.

The published paper is linked in README. Original labels, prompts, raw responses,
data license, sampling frame, model version and evaluation dates are unavailable
here. No numeric replication or causal claim is made.

## 2. Data contract

One row = one independent item, one ground-truth label, four condition outputs.
sample_id is a stable unique identifier. Run separately per model, task and
replicate. Do not assign new IDs to correlated repeated runs to evade validation.
Related claims from one event require a clustered model not implemented here.

CSV labels are read as strings; only empty/whitespace cells count as missing.
The full label universe must be explicit. Ordinal labels must be provided in
ascending semantic order; gaps in the observed labels must not collapse the scale.
All labels must be declared, including labels in rows later excluded from the
common cohort. Ground truth cannot be missing.

Available-case metrics use each condition's observed responses. Inferential
comparisons and common-case metrics use the same four-condition complete cohort.
Report both coverage and estimates; complete-case analysis does not correct
missing-not-at-random responses. No imputation is performed.

## 3. Metrics and inference

- Accuracy is the proportion of exact label matches.
- Macro-F1 averages over the entire declared label universe; classes with no
  support/prediction contribute zero. Compare studies only with the same universe.
- Cohen's κ measures chance-adjusted agreement. Linear/quadratic κ and Spearman
  require an explicit ordinal scale. Spearman measures association, not agreement.
- Undefined κ or correlation is null, not zero; missing bootstrap inference is
  also null/unavailable, not a negative finding.
- Let C_N, C_W, C_T, C_B be each item's binary correctness under the four conditions.
  The three planned comparisons are mean(C_j − C_N). Accuracy differences are
  proportions; multiply by 100 for percentage points.
- Each confidence interval resamples independent item-level contrasts with
  replacement (paired percentile bootstrap, 95%, default 5,000 draws). Seed is
  fixed and items are sorted by ID before resampling. Intervals are pointwise:
  they are not simultaneous intervals and may disagree with adjusted p-values.
- Exact McNemar uses the two-sided binomial test on discordant wins/losses with
  null probability 0.5. No discordance gives p=1. Holm correction covers all
  three planned baseline comparisons, not a selectively reported subset.
- Search main contrast: mean(((C_W − C_N) + (C_B − C_T))/2).
- Thinking main contrast: mean(((C_T − C_N) + (C_B − C_W))/2).
- Interaction: mean(C_B − C_W − C_T + C_N). This difference-in-differences is on
  the accuracy scale, not a logistic regression coefficient. Its range is [-2,2].
  The exploratory contrasts have pointwise bootstrap intervals, no extra p-tests.

Fewer than two common items disables paired inference. Even above that threshold,
small samples and degenerate empirical distributions make bootstrap intervals
unreliable; an interval of zero width does not prove population certainty.
These tests assume independent items, not independent conditions.

Method references:
[SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
[paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
[scikit-learn κ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html).

## 4. New experiment checklist

Before collecting new responses, freeze the task definition, sampling frame,
ground truth adjudication, label order, primary outcome and comparison family.
Use fresh conversations and randomize condition order; log model/version,
temperature, collection time, prompt hash, tool settings, evidence URLs, raw
response, parser version, failures and latency. Preserve refusal/missing outputs.
Do not tune prompts on the held-out evaluation set.

For original-paper numeric reproduction, obtain the original raw data and its
license, exact condition settings and prompts, ground-truth codebook, model
snapshot and sampling parameters, preprocessing rules, and expected tables.
This repository currently reproduces the new offline calculation pipeline only.

Future work, not implemented: clustered repeated-run bootstrap, topic/time
generalization, multi-model comparisons, calibration from genuine probabilities,
and evidence/cost/latency tradeoffs. Pre-register the additional comparisons and
avoid interpreting search or thinking effects as causal without a valid design.
