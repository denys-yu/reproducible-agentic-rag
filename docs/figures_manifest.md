# Figures manifest — revised article

Every number in every figure and table below is READ from the aggregator report files or
recomputed from `runs/` and `runs2/` through the aggregator's own functions in `src/metrics.py`.
Nothing is hardcoded. The values quoted in the task specification are treated as ASSERTIONS the
build script checks after reading; a mismatch aborts the build and prints both values.

Figures are rendered by `src/figures.py` through the label module `src/labels.py`, in Ukrainian
and English, as 600-dpi PNG and vector PDF, named `<figure_id>_<lang>.<ext>` in `figures/`.

Read-only on `runs/`, `runs2/` and `docs/reports/`. No API calls.

---

## Table T1 — `template_echo`: the format appendix echoed as data

A table, not a figure. Five rows over three constant columns; a chart would add decoration and
no information.

**Source:** `runs2/4omini_ablation/free_run{1..5}/run_manifest.jsonl`, grade node,
`parse_status == "partial"`.
**Fields read:** `question_id`, `run_id`, `node`, `parse_status`, `parsed.scope`,
`parsed.confidence`, `parsed.needs_more_context`, `raw_response`, `format_suffix_nodes`.

| question_id | runs affected | run ids | field lost | fields retained | value carried in the echoed line |
|---|---:|---|---|---|---|
| `5a836a2d554299334474600f` | 5 of 5 | r1, r2, r3, r4, r5 | `scope` | `confidence`, `needs_more_context` | `partial` |
| `5abd245555429924427fcf0f` | 3 of 5 | r2, r4, r5 | `scope` | `confidence`, `needs_more_context` | `none` |
| `5ade0a275542997545bbbe2d` | 5 of 5 | r1, r2, r3, r4, r5 | `scope` | `confidence`, `needs_more_context` | `partial` |
| `5ae28058554299495565da90` | 5 of 5 | r1, r2, r3, r4, r5 | `scope` | `confidence`, `needs_more_context` | `none` |
| `5ae604ec5542996de7b71af9` | 2 of 5 | r1, r5 | `scope` | `confidence`, `needs_more_context` | `none`, `partial` |

- Records: **20** partial grade calls over **5** distinct questions.
- `scope` lost in **20 of 20**; `confidence` and `needs_more_context` retained in **20 of 20**.
- Placeholder label emitted: `field_name` (20 of 20).

**What happened.** The format appendix (`src/prompts.py:70-76`, `FREE_FORMAT_SUFFIX`) instructs the
model to write one line per field "in this exact form", and then shows the form on its own line:

```
field_name: value
```

In these 20 calls the model reproduced that line **literally**, emitting `field_name:` as the label
instead of substituting `scope`. A representative raw output, verbatim:

```
field_name: partial
confidence: medium
needs_more_context: yes
```

The parser found no `scope:` line, so `parsed.scope` is `null` and
`provenance.classify_parse_status` returns `partial` (1-2 of 3 grade fields present; `ok` requires
all three, `unparsed` means none).

**Why this belongs in the results, not a footnote.** The graded judgement was *correct* in every
one of the 20 cases — `partial` and `none` are sensible scope values, carried intact in the echoed
line — and was destroyed purely at the label. This is not a reasoning failure and not a decoding
failure; it is an instruction-following failure in which the instruction meant to prevent format
error *is* the format error. Three of the five questions echo in all five runs, so the behaviour is
deterministic per question rather than sampling noise, consistent with temperature 0.

**Assertions checked:** 20 partial records; 5 distinct questions; `scope` absent in 20 of 20;
`confidence` and `needs_more_context` present in 20 of 20; echoed label is `field_name` in 20 of 20;
0 partial records in the published series (3399 records); every partial record carries
`format_suffix_nodes == ["grade"]`.

### The appendix is necessary but not sufficient

The failure appears only where the appendix is wired in, and `_SUFFIXED` in `src/prompts.py` wires
it to exactly one triple:

```python
_SUFFIXED = frozenset({(Condition.ABLATION, Arm.FREE.value, "grade")})
```

Confirmed against the manifests:

| cell | arm | node | `format_suffix_nodes` | n | partials |
|---|---|---|---|---:|---:|
| 4omini_ablation | free | **grade** | `["grade"]` | 750 | **20** |
| 4omini_ablation | free | rewrite, synthesize | `["grade"]` | 962 | 0 |
| luna_ablation | free | **grade** | `["grade"]` | 750 | **0** |
| luna_ablation | free | rewrite, synthesize | `["grade"]` | 1014 | 0 |
| 4omini_published | enum, free | all | `null` | 3399 | 0 |
| luna_published | enum, free | all | `[]` | 3307 | 0 |

**gpt-5.6-luna received the byte-identical appendix at the same node and produced zero echoes**
(0 of 750 grade calls). The appendix is therefore *necessary* for this failure — it occurs nowhere
without it, in the 8682 records that either receive no appendix (6706 published) or sit at a node
that never receives one (1976 ablation rewrite and synthesize calls) — but it is *not sufficient*:
the echo is specific to gpt-4o-mini. The claim the results should carry
is that adding a prose format contract introduces a failure mode that did not exist before it, on
at least one model, and not that the contract causes the failure on any model that receives it.

The appendix is byte-identical across the two models by construction — it is the single
module-level constant `FREE_FORMAT_SUFFIX`, not a per-model template — and that it is actually
applied that way is enforced rather than assumed: `src/preflight.py:138-142` asserts
`(free, grade) == (enum, grade) + FREE_FORMAT_SUFFIX` byte-exact, the check immediately after it
asserts the suffix does not leak past the grade node, and `src/llm.py:549-556` refuses to log a
record whose system prompt does not match the wiring its condition declares.

---

## Figures

Built by `python -m src.figures --revised` (or `src.figures.build_revised`), which reads
`src/figure_data.py` and renders through `src/labels.py`. Each figure is written in four files:
Ukrainian and English, PNG at 600 dpi and vector PDF.

**Build properties asserted by `tests/test_figures_determinism.py` (22 tests):**

- Two consecutive builds are **byte-identical** in all 24 files. `_save` suppresses the PDF
  `/CreationDate`, `Creator` and `Producer` and the PNG `Software` chunk, which otherwise stamp a
  timestamp into every file; the font is pinned to DejaVu Sans rather than inherited.
- Both language variants of each figure contain the **same number of text elements** (277 per
  language across the six figures).
- No Ukrainian figure carries untranslated English outside the identifier and model-name allowlist
  in `src/labels.py`.
- Ukrainian prints a decimal comma and English a decimal point on every drawn value, axis tick
  labels included.

**Grayscale:** no colour anywhere. All fills are white; series separate by hatch
(`...` no format declared, `///` format appendix, `\\\` schema, `xxx` degenerate/unstable) and by
marker shape (square = EM, triangle = token-level F1), with direct labelling instead of legends.
Every value is printed on or beside its mark.

### Figure 1 — `route_stability`

**Route stability across three format states**

- **Files:** `route_stability_uk.png` · `route_stability_uk.pdf` · `route_stability_en.png` · `route_stability_en.pdf` (in `figures/`)
- **Source:** `report_series2_agg.txt` section [2] — route stability, flip count, rewrite_rate, DEGENERATE flag
- **Fields read:** `route_stability`, `flips`, `rewrite_rate`, degeneracy flag (6 cells)
- **Assertions checked:** 18

  | check | asserted value |
  |---|---:|
  | `route_stab\|4omini_ablation\|free` | 0.9333 |
  | `route_stab\|4omini_published\|enum` | 0.9733 |
  | `route_stab\|4omini_published\|free` | 0.8333 |
  | `route_stab\|luna_ablation\|free` | 0.9267 |
  | `route_stab\|luna_published\|enum` | 0.9200 |
  | `route_stab\|luna_published\|free` | 0.8867 |
  | `flips\|4omini_ablation\|free` | 10 |
  | `flips\|4omini_published\|enum` | 4 |
  | `flips\|4omini_published\|free` | 25 |
  | `flips\|luna_ablation\|free` | 11 |
  | `flips\|luna_published\|enum` | 12 |
  | `flips\|luna_published\|free` | 17 |
  | `rewrite\|4omini_ablation\|free` | 0.2827 |
  | `rewrite\|4omini_published\|enum` | 0.2373 |
  | `rewrite\|4omini_published\|free` | 0.2947 |
  | `rewrite\|luna_ablation\|free` | 0.3520 |
  | `rewrite\|luna_published\|enum` | 0.3373 |
  | `rewrite\|luna_published\|free` | 0.0720 |

### Figure 2 — `format_guess_stability`

**Format-guess stability**

- **Files:** `format_guess_stability_uk.png` · `format_guess_stability_uk.pdf` · `format_guess_stability_en.png` · `format_guess_stability_en.pdf` (in `figures/`)
- **Source:** `report_series2_agg.txt` section [1]; decoding provenance recomputed from `runs2/luna_published/free_run{1..5}/run_manifest.jsonl`
- **Fields read:** histogram over 0-5 `json_object` runs, shape-SPLIT count and share, 4o-mini shape-stable count; `temperature`, `top_p`, `seed`, `prompt_sha256`
- **Assertions checked:** 14

  | check | asserted value |
  |---|---:|
  | `fmt_4omini_stable` | 150 |
  | `fmt_hist\|0` | 9 |
  | `fmt_hist\|1` | 7 |
  | `fmt_hist\|2` | 8 |
  | `fmt_hist\|3` | 11 |
  | `fmt_hist\|4` | 26 |
  | `fmt_hist\|5` | 89 |
  | `fmt_split_n` | 52 |
  | `fmt_split_share` | 0.3467 |
  | `fmt_total` | 150 |
  | `decode_prompt_identical_questions` | 150 |
  | `decode_seed` | 42 |
  | `decode_temperature` | 0.0000 |
  | `decode_top_p` | 1 |

### Figure 3 — `silent_failure`

**Silent failure mechanism at the grade node**

- **Files:** `silent_failure_uk.png` · `silent_failure_uk.pdf` · `silent_failure_en.png` · `silent_failure_en.pdf` (in `figures/`)
- **Source:** `unparsed_rate` and the partial count recomputed via `src.metrics.unparsed_rate` over `src.metrics.load_manifests`; `rewrite_rate` read from section [2] **and** cross-checked against a fresh `src.metrics.rewrite_rate`
- **Fields read:** `parse_status` per grade call (`ok` / `partial` / `unparsed`), `rewrote` per pipeline run
- **Assertions checked:** 13

  | check | asserted value |
  |---|---:|
  | `unparsed\|4omini_ablation\|free` | 0.0000 |
  | `unparsed\|4omini_published\|enum` | 0.0000 |
  | `unparsed\|4omini_published\|free` | 0.0013 |
  | `unparsed\|luna_ablation\|free` | 0.0000 |
  | `unparsed\|luna_published\|enum` | 0.0000 |
  | `unparsed\|luna_published\|free` | 0.8067 |
  | `partials\|4omini_ablation\|free` | 20 |
  | `rewrite\|4omini_ablation\|free` | 0.2827 |
  | `rewrite\|4omini_published\|enum` | 0.2373 |
  | `rewrite\|4omini_published\|free` | 0.2947 |
  | `rewrite\|luna_ablation\|free` | 0.3520 |
  | `rewrite\|luna_published\|enum` | 0.3373 |
  | `rewrite\|luna_published\|free` | 0.0720 |

### Figure 4 — `coverage_collapse`

**Why coverage must be reported**

- **Files:** `coverage_collapse_uk.png` · `coverage_collapse_uk.pdf` · `coverage_collapse_en.png` · `coverage_collapse_en.pdf` (in `figures/`)
- **Source:** `report_series2_agg.txt` section [3], endpoint `grade.needs_more_context`
- **Fields read:** `EMA`, `coverage`, `usable_agreement` (6 cells)
- **Assertions checked:** 18

  | check | asserted value |
  |---|---:|
  | `ema\|4omini_ablation\|free` | 0.9627 |
  | `ema\|4omini_published\|enum` | 0.9893 |
  | `ema\|4omini_published\|free` | 0.9213 |
  | `ema\|luna_ablation\|free` | 0.9627 |
  | `ema\|luna_published\|enum` | 0.9600 |
  | `ema\|luna_published\|free` | 0.9571 |
  | `cov\|4omini_ablation\|free` | 1 |
  | `cov\|4omini_published\|enum` | 1 |
  | `cov\|4omini_published\|free` | 0.9973 |
  | `cov\|luna_ablation\|free` | 1 |
  | `cov\|luna_published\|enum` | 1 |
  | `cov\|luna_published\|free` | 0.1113 |
  | `usable\|4omini_ablation\|free` | 0.9627 |
  | `usable\|4omini_published\|enum` | 0.9893 |
  | `usable\|4omini_published\|free` | 0.9189 |
  | `usable\|luna_ablation\|free` | 0.9627 |
  | `usable\|luna_published\|enum` | 0.9600 |
  | `usable\|luna_published\|free` | 0.1066 |

### Figure 5 — `field_collapse`

**Field collapse between synthesize.scope and synthesize.confidence**

- **Files:** `field_collapse_uk.png` · `field_collapse_uk.pdf` · `field_collapse_en.png` · `field_collapse_en.pdf` (in `figures/`)
- **Source:** `report_series2_agg.txt` section [5]
- **Fields read:** bijection `breaks` and the denominator `n` (4 cells)
- **Assertions checked:** 5

  | check | asserted value |
  |---|---:|
  | `breaks_n` | 750 |
  | `breaks\|4omini_ablation\|free` | 5 |
  | `breaks\|4omini_published\|enum` | 0.0000 |
  | `breaks\|luna_ablation\|free` | 65 |
  | `breaks\|luna_published\|enum` | 44 |

### Figure 6 — `length_artifact`

**Answer length against quality**

- **Files:** `length_artifact_uk.png` · `length_artifact_uk.pdf` · `length_artifact_en.png` · `length_artifact_en.pdf` (in `figures/`)
- **Source:** `report_final_analysis.txt` section [3]
- **Fields read:** mean answer `tokens`, `EM`, token-level `F1` (6 cells)
- **Assertions checked:** 18

  | check | asserted value |
  |---|---:|
  | `tokens\|4omini_ablation\|free` | 11.6500 |
  | `tokens\|4omini_published\|enum` | 14.5900 |
  | `tokens\|4omini_published\|free` | 11.8600 |
  | `tokens\|luna_ablation\|free` | 9.6000 |
  | `tokens\|luna_published\|enum` | 13.9300 |
  | `tokens\|luna_published\|free` | 9.5900 |
  | `em\|4omini_ablation\|free` | 0.1360 |
  | `em\|4omini_published\|enum` | 0.0280 |
  | `em\|4omini_published\|free` | 0.1360 |
  | `em\|luna_ablation\|free` | 0.2240 |
  | `em\|luna_published\|enum` | 0.0680 |
  | `em\|luna_published\|free` | 0.2067 |
  | `f1\|4omini_ablation\|free` | 0.3659 |
  | `f1\|4omini_published\|enum` | 0.2580 |
  | `f1\|4omini_published\|free` | 0.3656 |
  | `f1\|luna_ablation\|free` | 0.4577 |
  | `f1\|luna_published\|enum` | 0.3035 |
  | `f1\|luna_published\|free` | 0.4390 |

<!-- total assertions across the six figures: 86 -->
