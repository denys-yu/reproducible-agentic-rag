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

## 6. Amendment, 2026-08-24, after the smoke run and before the second series

The smoke run revealed that agreement metrics were computed over records whose output failed
to parse. Two failed parses yield None on both sides and were scored as agreement. A model
that always fails would appear perfectly stable, which inverts the quantity this paper is
about. The following conventions are fixed now, before the second series is executed.

6.1 None is not a category. For every agreement endpoint, a pair in which either side is None
is excluded from kappa, EMA and TAR.

6.2 Coverage, the number of pairs actually scored over the number available, is reported next
to every agreement figure. A figure with coverage below 0.5 is marked in every table and is
not used to support a claim about stability.

6.3 unparsed_rate becomes a first-class descriptive endpoint, reported per model, condition,
arm and node. It is not part of the corrected family.

6.4 Route on missing signal. When needs_more_context is None because the grade output did not
parse, the route actually taken is recorded and reported as route_on_missing_signal. Smoke
evidence: gpt-5.6-luna, published prompts, free arm, 23 of 30 grade calls unparsed and zero
rewrites of 30, against 7 of 30 in the enum arm.

6.5 Forecast F3 is provisionally confirmed on smoke data: luna free arm, published prompts,
grade node, unparsed rate 0.767 against a registered threshold of 0.5. The same cell under
ablation parsed 30 of 30, which isolates the effect to the absence of a declared format
rather than to the model. Confirmation on the full series is still required.

6.6 Composite reporting. Every agreement figure is accompanied by usable_agreement =
coverage x EMA, the probability that a question yields a scorable pair that also agrees.
Applied to EMA and TAR only, not to kappa, which is chance-corrected and does not compose.
Rationale: under 6.1 an arm that fails to parse most of the time can show EMA 1.000 on the
few pairs it does produce, which reads as higher stability than an arm that answers every
time. Smoke evidence: luna, published, free, grade shows EMA 1.000 at coverage 0.133 against
EMA 0.933 at coverage 1.000 in the enum arm.

6.7 Forecast F5. The luna free arm under published prompts will remain below 0.5 coverage on
the full series. At a per-call parse rate near 0.233 and five runs, a question yields at least
one scorable pair with probability near 0.33. Claims about that cell will therefore rest on
unparsed_rate, usable_agreement and routing, not on kappa. Registered before execution.

6.8 Degenerate routing. route_stability remains the pre-registered primary endpoint and is
reported as registered. It is always reported beside rewrite_rate. A cell whose rewrite_rate
falls below 0.10 or rises above 0.90 is marked as degenerate routing, and its stability figure
does not support any claim about control quality, because a router that almost never fires is
trivially stable. Evidence forcing this amendment: on the second series the luna free arm under
published prompts scores route_stability 0.8867 at rewrite_rate 0.072, above the 0.8333 of the
working 4o-mini free arm at rewrite_rate 0.2947. The higher figure reflects collapse, not
control. This amendment is post hoc with respect to the second series and is recorded as such.

6.9 Forecast F4 is at risk and will be adjudicated on both readings. Read as rewrite_rate, the
enum-free direction reverses between models: on 4o-mini free exceeds enum, on luna enum exceeds
free. Read as route_stability, the direction holds in both. Both readings are reported and the
forecast is scored as partially refuted rather than reinterpreted.