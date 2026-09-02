"""Internationalisation (i18n) strings for the publication figures.

Every human-readable string that appears on a figure — titles, axis labels, tick/category labels,
legend entries, on-bar annotations, direct series labels, and footnotes — lives here, keyed first
by language (``"uk"`` / ``"en"``) and then by a stable string key. ``src.figures`` looks strings up
by key so that a single run renders both language variants with byte-identical layout and only the
text differing. No string is ever baked into the drawing code.

``decimal_sep`` is the locale's decimal separator; ``src.figures`` uses it to format every number
drawn on a figure (bar values, p-values, coverage, counts) without relying on the ``locale`` module.

Two families of keys live here. ``fig1_``/``fig2_`` belong to the original paper figures
(``kappa_by_field``, ``answer_quality``). ``rf1_``…``rf6_`` and the shared ``state_``/``cell_``
keys belong to the revised article's six figures; they are deliberately a separate key space so
that adding the revised figures cannot disturb the originals.

Fixed terminology for the Ukrainian variants — do not paraphrase:
    route stability -> стабільність маршруту      rewrite rate -> частота переписування
    unparsed rate   -> частка нерозібраних        coverage     -> покриття
    usable agreement-> придатна згода             prose format description ->
                                                      словесний опис формату
    no format declared -> формат не проголошено   schema       -> схема
    question -> запитання                          runs         -> запусків
    degenerate routing -> вироджена маршрутизація  answer length -> довжина відповіді
    tokens   -> токенів

Field identifiers from the code (``grade.needs_more_context``, ``synthesize.scope``,
``synthesize.confidence``, ``enum``, ``free``) and model names stay untranslated in both variants.
Code identifiers never appear as reader-facing labels: ``json_object`` is drawn as "об'єкт JSON" /
"a JSON object", and ``labeled_lines`` as "позначені рядки" / "labeled lines".
"""

from __future__ import annotations

LABELS: dict[str, dict[str, str]] = {
    "uk": {
        # ── Figure 1 (original): kappa_by_field ──────────────────
        "fig1_y_axis": "Середня попарна каппа Коена (κ)",
        "fig1_legend_enum": "структурований вивід",
        "fig1_legend_free": "вільний текст",
        "fig1_cat_grade_conf": "Впевненість\n(оцінювання)",
        "fig1_cat_grade_scope": "Охоплення\nконтексту",
        "fig1_cat_grade_needs": "Потреба\nв пошуку",
        "fig1_cat_syn_conf": "Впевненість\n(синтез)",
        "fig1_cat_syn_scope": "Охоплення\n(синтез)",
        "fig1_cat_answer": "Фінальна\nвідповідь",
        "fig1_annot_p_prefix": "pH=",
        "fig1_footnote": ("pH — p-значення парного критерію Вілкоксона для попарних "
                          "збігів після поправки Голма на шість кінцевих точок."),
        # ── Figure 2 (original): answer_quality ──────────────────
        "fig2_y_axis": "Середнє значення метрики",
        "fig2_legend_enum": "структурований вивід",
        "fig2_legend_free": "вільний текст",
        "fig2_m_em": "Exact Match",
        "fig2_m_f1": "Токен-рівневий F1",
        "fig2_m_containment": "Входження еталонної\nвідповіді",
        "fig2_annot_p_prefix": "p=",
        "fig2_annot_descriptive": "описове\nпорівняння",
        "fig2_footnote": ("p — парний критерій Вілкоксона на рівні запитань; "
                          "для Exact Match p не розраховували."),

        # ── shared: the three format states ──────────────────────
        "state_no_format": "формат не проголошено",
        "state_appendix": "словесний опис формату",
        "state_schema": "схема",

        # ── RF1: route_stability ─────────────────────────────────
        "rf1_title": "Стабільність маршруту у трьох станах формату",
        "rf1_y_axis": "Стабільність маршруту\n(однаковий маршрут у всіх 5 запусках)",
        "rf1_degenerate_note": "Вироджена маршрутизація: частота переписування 0,072",
        "rf1_baseline_note": ("Ланцюг із трьох станів читається лише на gpt-4o-mini:\n"
                              "gpt-5.6-luna не має придатного базового стану."),

        # ── RF2: format_guess_stability ──────────────────────────
        "rf2_title": "Стабільність вгадування формату",
        "rf2_x_axis": "Запусків (із 5), що повернули об'єкт JSON",
        "rf2_y_axis": "Запитань",
        "rf2_unstable_label": "нестабільна зона (1–4)",
        "rf2_omini_label": ("gpt-4o-mini: 150 зі 150 запитань\n"
                            "стабільні на позначених рядках у всіх 5 запусках"),
        "rf2_luna_label": "gpt-5.6-luna, формат не проголошено",
        "rf2_footnote": ("Декодування: temperature 0, top_p 1, seed 42. Побайтово однакова "
                         "вказівка для моделі в усіх 5 запусках (150 зі 150 запитань)."),

        # ── RF3: silent_failure ──────────────────────────────────
        "rf3_title": "Механізм мовчазної відмови керування на вузлі оцінювання",
        "rf3_y_axis_top": "Частка нерозібраних",
        "rf3_y_axis_bottom": "Частота переписування",
        "rf3_partial_note": ("20 часткових розбирань (відлуння шаблону)\n"
                             "не входять до частки нерозібраних"),
        "rf3_panel_a": "а",
        "rf3_panel_b": "б",

        # ── RF4: coverage_collapse ───────────────────────────────
        "rf4_title": "Чому покриття потрібно повідомляти",
        "rf4_y_axis": "Згода за grade.needs_more_context",
        "rf4_series_ema": "EMA",
        "rf4_series_usable": "придатна згода",
        "rf4_coverage_prefix": "покриття ",
        "rf4_collapse_note": ("gpt-5.6-luna, формат не проголошено:\n"
                              "виглядає стабільним лише за EMA\nі руйнується за придатною згодою"),

        # ── RF5: field_collapse ──────────────────────────────────
        "rf5_title": "Руйнування полів: розриви бієкції\nміж synthesize.scope та synthesize.confidence",
        "rf5_y_axis": "Розривів бієкції (із 750)",

        # ── RF6: length_artifact ─────────────────────────────────
        "rf6_title": "Довжина відповіді проти якості",
        "rf6_x_axis": "Середня довжина відповіді (токенів)",
        "rf6_y_axis": "Значення метрики",
        "rf6_series_em": "EM",
        "rf6_series_f1": "F1 на рівні токенів",
        "rf6_footnote": ("Гілка зі схемою довша й отримує нижчі оцінки на обох моделях: "
                         "дефіцит є артефактом довжини."),

        # ── shared ───────────────────────────────────────────────
        "decimal_sep": ",",
        "of_connector": " зі ",
    },
    "en": {
        # ── Figure 1 (original): kappa_by_field ──────────────────
        "fig1_y_axis": "Mean pairwise Cohen's kappa (κ)",
        "fig1_legend_enum": "structured output",
        "fig1_legend_free": "free text",
        "fig1_cat_grade_conf": "Confidence\n(grading)",
        "fig1_cat_grade_scope": "Context\nscope",
        "fig1_cat_grade_needs": "Re-retrieval\nneed",
        "fig1_cat_syn_conf": "Confidence\n(synthesis)",
        "fig1_cat_syn_scope": "Scope\n(synthesis)",
        "fig1_cat_answer": "Final\nanswer",
        "fig1_annot_p_prefix": "pH=",
        "fig1_footnote": ("pH — paired Wilcoxon p-value for pairwise matches after "
                          "Holm correction over six endpoints."),
        # ── Figure 2 (original): answer_quality ──────────────────
        "fig2_y_axis": "Mean metric value",
        "fig2_legend_enum": "structured output",
        "fig2_legend_free": "free text",
        "fig2_m_em": "Exact Match",
        "fig2_m_f1": "Token-level F1",
        "fig2_m_containment": "Gold-answer\ncontainment",
        "fig2_annot_p_prefix": "p=",
        "fig2_annot_descriptive": "descriptive\ncomparison",
        "fig2_footnote": ("p — paired Wilcoxon test at question level; p was not "
                          "computed for Exact Match."),

        # ── shared: the three format states ──────────────────────
        "state_no_format": "no format declared",
        "state_appendix": "prose format description",
        "state_schema": "schema",

        # ── RF1: route_stability ─────────────────────────────────
        "rf1_title": "Route stability across three format states",
        "rf1_y_axis": "Route stability\n(identical route in all 5 runs)",
        "rf1_degenerate_note": "Degenerate routing: rewrite rate 0.072",
        "rf1_baseline_note": ("The three-state chain reads only on gpt-4o-mini:\n"
                              "gpt-5.6-luna has no valid baseline."),

        # ── RF2: format_guess_stability ──────────────────────────
        "rf2_title": "Format-guess stability",
        "rf2_x_axis": "Runs (of 5) that returned a JSON object",
        "rf2_y_axis": "Questions",
        "rf2_unstable_label": "unstable region (1–4)",
        "rf2_omini_label": ("gpt-4o-mini: 150 of 150 questions\n"
                            "stable at labeled lines in all 5 runs"),
        "rf2_luna_label": "gpt-5.6-luna, no format declared",
        "rf2_footnote": ("Decoding: temperature 0, top_p 1, seed 42. A byte-identical model "
                         "instruction in all 5 runs (150 of 150 questions)."),

        # ── RF3: silent_failure ──────────────────────────────────
        "rf3_title": "Silent control failure mechanism at the grading node",
        "rf3_y_axis_top": "Unparsed rate",
        "rf3_y_axis_bottom": "Rewrite rate",
        "rf3_partial_note": ("20 partial parses (template echo)\n"
                             "are not counted in the unparsed rate"),
        "rf3_panel_a": "a",
        "rf3_panel_b": "b",

        # ── RF4: coverage_collapse ───────────────────────────────
        "rf4_title": "Why coverage must be reported",
        "rf4_y_axis": "Agreement on grade.needs_more_context",
        "rf4_series_ema": "EMA",
        "rf4_series_usable": "usable agreement",
        "rf4_coverage_prefix": "coverage ",
        "rf4_collapse_note": ("gpt-5.6-luna, no format declared:\n"
                              "reads as stable on EMA alone\nand collapses on usable agreement"),

        # ── RF5: field_collapse ──────────────────────────────────
        "rf5_title": "Field collapse: bijection breaks\nbetween synthesize.scope and synthesize.confidence",
        "rf5_y_axis": "Bijection breaks (of 750)",

        # ── RF6: length_artifact ─────────────────────────────────
        "rf6_title": "Answer length against quality",
        "rf6_x_axis": "Mean answer length (tokens)",
        "rf6_y_axis": "Metric value",
        "rf6_series_em": "EM",
        "rf6_series_f1": "Token-level F1",
        "rf6_footnote": ("The schema arm is longer and scores lower on both models: "
                         "the deficit is a length artifact."),

        # ── shared ───────────────────────────────────────────────
        "decimal_sep": ".",
        "of_connector": " of ",
    },
}

#: Output file stem per figure (the ``<figure_id>`` in ``<figure_id>_<lang>.<ext>``).
#: ``fig1``/``fig2`` are the original paper figures; ``rf1``…``rf6`` the revised article's.
FIGURE_IDS: dict[str, str] = {
    "fig1": "kappa_by_field",
    "fig2": "answer_quality",
    "rf1": "route_stability",
    "rf2": "format_guess_stability",
    "rf3": "silent_failure",
    "rf4": "coverage_collapse",
    "rf5": "field_collapse",
    "rf6": "length_artifact",
}

#: The revised article's figures, in build order.
REVISED_FIGURE_KEYS: tuple[str, ...] = ("rf1", "rf2", "rf3", "rf4", "rf5", "rf6")

#: Languages rendered on every run, in output order.
LANGS: tuple[str, ...] = ("uk", "en")

#: Strings allowed to appear untranslated in a Ukrainian figure: code identifiers and model names.
IDENTIFIER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "grade.needs_more_context",
        "synthesize.scope",
        "synthesize.confidence",
        "enum",
        "free",
        "grade",
        # A format name the article keeps in Latin script by house style, not an identifier.
        "JSON",
        "EMA",
        "EM",
        "F1",
        # Decoding parameters, which are literal `request_params` keys in the manifests.
        "temperature",
        "top_p",
        "seed",
    }
)

#: Model names, unchanged in both variants.
MODEL_NAMES: frozenset[str] = frozenset({"gpt-4o-mini", "gpt-5.6-luna"})
