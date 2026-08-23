# Pre-registration, second experimental series and ablation
Date: 2026-08-23. Committed before any second-series execution.

## 1. Endpoint family

Primary endpoint (single, no correction):
- route_stability: per question, whether the rewrite decision is identical across all 5 runs.
  Published values: enum 0.9733 (4 switches of 150), free 0.8333 (25 of 150).
  Test: McNemar exact test on the 150 paired questions.

Secondary family, Holm correction over 5 endpoints:
- grade.needs_more_context
- grade.confidence
- grade.scope
- synthesize.scope
- answer.normalized

Removed from the family: synthesize.confidence. Reason: in the enum arm it is an exact
bijection with synthesize.scope across all 750 calls (full→high 709, none→low 21,
partial→medium 20). The two are one variable under two names. In the free arm the link
breaks in 6 of 744 calls. Keeping both would apply the correction to a duplicate.
Recomputing the published series with a family of 5 changes no conclusion:
grade.scope moves from 0.123 to 0.092, still non-significant. All other verdicts unchanged.
synthesize.confidence is still reported, descriptively, with the bijection disclosed.

Descriptive, no correction: query-text stability, document-set identity at synthesis,
system_fingerprint association, EM, token-level F1, gold-span containment,
output_shape distribution, unparsed rate.

## 2. Agreement convention

Agreement across 5 runs is the mean pairwise Cohen's kappa over the 10 unordered run pairs,
computed per question and aggregated. Arms are compared with a paired Wilcoxon test over
questions. Identical to the published series.

Skewed and degenerate fields. If a field takes one category in 95 percent or more of calls
within an arm, kappa is unstable and is reported together with exact match agreement (EMA),
with EMA as the primary quantity for that field. If a field is fully degenerate in a run,
kappa is undefined and only EMA is reported, marked as such in the table. The 95 percent
threshold is fixed here, before the data exist.

## 3. Conditions

| Condition | Format communicated | Decoding constrained |
|---|---|---|
| free, published prompts | no | no |
| free + format appendix on the grade node only (ablation) | yes | no |
| enum, schema | yes | yes |

The appendix is never attached to the synthesize node. Pilot run 3 attached it to both and
"add no commentary" cut free-arm answers to 3.72 tokens against 14.83, making F1, EM and
answer agreement meaningless. This is now asserted at run start and logged per record in
format_suffix_nodes.

## 4. Forecasts

F1. On gpt-5.6-luna the enum arm will show higher agreement than the free arm on
grade.needs_more_context, in the same direction as the published series.

F2. The ablation gap will be smaller than the published gap on grade.needs_more_context and
may fail to reach significance. Basis: published series without any format instruction gives
EMA 0.989 against 0.919, gap 0.071, while pilot run 3 with a format instruction gives
1.000 against 0.973, gap 0.027. Different sample sizes, so this is an indication, not proof.

F3. The free arm on luna will produce json_object output shape where 4o-mini produced
labeled_lines in 750 of 750 grade calls, and its unparsed rate will exceed 0.5.

F4. Absolute routing thresholds will differ between the two models. The direction of the
enum-free difference will not.

A failed forecast is reported as a failed forecast, not silently reframed.

## 5. Frozen inputs

Question set: pins/published_question_ids.json, 150 ids,
sha256 498ec7b8b03177bf6ec2ef7804977289589b0448eed4f3cfec11784fbd7fee47.
Smoke subset: pins/smoke_question_ids.json, first 15 sorted ids.
Corpus: chroma collection hotpotqa_distractor, 1500 documents,
digest 5c4a72d6cc2c56b03d9a4426d944ab06fd671ac40d14f2049daf60ff284ae7b3,
text-embedding-3-small, top_k 4.
The sampler is not called again. The published series is not re-run.