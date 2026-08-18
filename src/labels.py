"""Internationalisation (i18n) strings for the publication figures.

Every human-readable string that appears on a figure — axis labels, tick/category labels,
legend entries, on-bar annotations, and footnotes — lives here, keyed first by language
(``"uk"`` / ``"en"``) and then by a stable string key. ``src.figures`` looks strings up by
key so that a single run renders both language variants with byte-identical layout and only
the text differing.

``decimal_sep`` is the locale's decimal separator; ``src.figures`` uses it to format every
number drawn on a figure (bar values and p-values) without relying on the ``locale`` module.
"""

from __future__ import annotations

LABELS: dict[str, dict[str, str]] = {
    "uk": {
        # ── Figure 1: kappa_by_field ─────────────────────────────
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
        # ── Figure 2: answer_quality ─────────────────────────────
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
        # ── shared ───────────────────────────────────────────────
        "decimal_sep": ",",
    },
    "en": {
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
        "decimal_sep": ".",
    },
}

# Output file stem per figure (the ``<figure_id>`` in ``<figure_id>_<lang>.<ext>``).
FIGURE_IDS: dict[str, str] = {"fig1": "kappa_by_field", "fig2": "answer_quality"}

# Languages rendered on every run, in output order.
LANGS: tuple[str, ...] = ("uk", "en")
