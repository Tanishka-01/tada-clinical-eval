import ast
import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_METRICS = ["cer", "rtf", "speaker_similarity", "pas_score", "cta_score", "ecs"]


def _bold(val: str) -> str:
    return f"\\textbf{{{val}}}"


def generate_main_results_table(df: pd.DataFrame) -> str:
    models = df["model_name"].unique().tolist()
    available = [m for m in NUMERIC_METRICS if m in df.columns]

    header = " & ".join(["Metric"] + models) + " \\\\"
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Main Evaluation Results: Mean $\\pm$ Std across all sentences}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{l" + "c" * len(models) + "}",
        "\\hline",
        header,
        "\\hline",
    ]

    for metric in available:
        means = {}
        stds = {}
        for model in models:
            col = pd.to_numeric(df[df["model_name"] == model][metric], errors="coerce").dropna()
            if len(col) > 0:
                means[model] = float(col.mean())
                stds[model] = float(col.std())
            else:
                means[model] = None
                stds[model] = None

        lower_is_better = metric in ["cer", "rtf"]
        valid = {k: v for k, v in means.items() if v is not None}
        best = (
            min(valid, key=valid.get) if lower_is_better else max(valid, key=valid.get)
        ) if valid else None

        cells = []
        for model in models:
            if means[model] is None:
                cells.append("N/A")
            else:
                cell = f"{means[model]:.3f} $\\pm$ {stds[model]:.3f}"
                if model == best:
                    cell = _bold(cell)
                cells.append(cell)

        lines.append(metric.upper() + " & " + " & ".join(cells) + " \\\\")

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def generate_category_table(df: pd.DataFrame) -> str:
    categories = df["category"].unique().tolist() if "category" in df.columns else []
    models = df["model_name"].unique().tolist()
    key_metric = next((m for m in NUMERIC_METRICS if m in df.columns), None)
    if key_metric is None:
        return "% No metrics available"

    header = " & ".join(["Category"] + [f"{m}\\n{key_metric.upper()}" for m in models]) + " \\\\"
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Category Breakdown ({key_metric.upper()} per domain)}}",
        "\\label{tab:category}",
        "\\begin{tabular}{l" + "c" * len(models) + "}",
        "\\hline",
        header,
        "\\hline",
    ]

    for cat in categories:
        cells = [cat.title()]
        for model in models:
            sub = df[(df["model_name"] == model) & (df["category"] == cat)]
            col = pd.to_numeric(sub[key_metric], errors="coerce").dropna()
            cells.append(f"{col.mean():.3f}" if len(col) > 0 else "N/A")
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def generate_cta_analysis_table(df: pd.DataFrame) -> str:
    if "cta_score" not in df.columns:
        return "% CTA not available"

    clinical_df = df[df["category"] == "clinical"] if "category" in df.columns else df
    models = df["model_name"].unique().tolist()

    term_errors: Dict[str, Dict[str, int]] = {}
    for _, row in clinical_df.iterrows():
        model = row["model_name"]
        incorrect = row.get("cta_incorrect_terms", [])
        if isinstance(incorrect, str):
            try:
                incorrect = ast.literal_eval(incorrect)
            except Exception:
                incorrect = []
        if not isinstance(incorrect, list):
            incorrect = []
        for term in incorrect:
            if term not in term_errors:
                term_errors[term] = {m: 0 for m in models}
            term_errors[term][model] = term_errors[term].get(model, 0) + 1

    if not term_errors:
        return "% No CTA error data available"

    sorted_terms = sorted(
        term_errors.items(), key=lambda x: sum(x[1].values()), reverse=True
    )[:15]

    header = " & ".join(["Medical Term"] + [f"{m}\\nErrors" for m in models]) + " \\\\"
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Most Commonly Hallucinated Medical Terms per Model}",
        "\\label{tab:cta_errors}",
        "\\begin{tabular}{l" + "c" * len(models) + "}",
        "\\hline",
        header,
        "\\hline",
    ]

    for term, errors in sorted_terms:
        cells = [term.replace("_", "\\_")] + [str(errors.get(m, 0)) for m in models]
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
