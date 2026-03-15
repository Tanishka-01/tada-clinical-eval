import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

NUMERIC_METRICS = ["cer", "rtf", "speaker_similarity", "pas_score", "cta_score", "ecs"]


def compute_summary_stats(df: pd.DataFrame) -> Dict:
    summary = {}
    models = df["model_name"].unique()
    for model in models:
        summary[model] = {}
        mdf = df[df["model_name"] == model]
        for metric in NUMERIC_METRICS:
            if metric not in mdf.columns:
                continue
            col = pd.to_numeric(mdf[metric], errors="coerce").dropna()
            if len(col) == 0:
                summary[model][metric] = {
                    "mean": None, "std": None, "median": None,
                    "min": None, "max": None, "ci_95": None, "n": 0,
                }
                continue
            mean = float(col.mean())
            std = float(col.std())
            n = len(col)
            ci_margin = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            summary[model][metric] = {
                "mean": mean,
                "std": std,
                "median": float(col.median()),
                "min": float(col.min()),
                "max": float(col.max()),
                "ci_95": (mean - ci_margin, mean + ci_margin),
                "n": n,
            }
    return summary


def compute_significance(df: pd.DataFrame, metric: str) -> Dict:
    result = {}
    models = df["model_name"].unique()
    groups = []
    group_names = []
    for model in models:
        col = pd.to_numeric(df[df["model_name"] == model][metric], errors="coerce").dropna()
        if len(col) > 0:
            groups.append(col.values)
            group_names.append(model)

    if len(groups) < 2:
        return {"error": "Not enough groups for significance testing"}

    if len(groups) >= 3:
        try:
            kw_stat, kw_pval = stats.kruskal(*groups)
            result["kruskal_wallis"] = {
                "statistic": float(kw_stat),
                "p_value": float(kw_pval),
                "significant": kw_pval < 0.05,
            }
        except Exception as e:
            result["kruskal_wallis"] = {"error": str(e)}

    tada_idx = next(
        (i for i, n in enumerate(group_names) if "tada" in n.lower()), None
    )
    if tada_idx is not None:
        tada_data = groups[tada_idx]
        pairwise = {}
        for i, (name, data) in enumerate(zip(group_names, groups)):
            if i == tada_idx:
                continue
            try:
                u_stat, p_val = stats.mannwhitneyu(
                    tada_data, data, alternative="two-sided"
                )
                pairwise[f"tada_vs_{name}"] = {
                    "u_statistic": float(u_stat),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                }
            except Exception as e:
                pairwise[f"tada_vs_{name}"] = {"error": str(e)}
        result["pairwise"] = pairwise

    return result


def compute_category_breakdown(df: pd.DataFrame) -> Dict:
    categories = df["category"].unique() if "category" in df.columns else []
    models = df["model_name"].unique()
    breakdown = {}
    for model in models:
        breakdown[model] = {}
        for cat in categories:
            sub = df[(df["model_name"] == model) & (df["category"] == cat)]
            breakdown[model][cat] = {}
            for metric in NUMERIC_METRICS:
                if metric not in sub.columns:
                    continue
                col = pd.to_numeric(sub[metric], errors="coerce").dropna()
                breakdown[model][cat][metric] = float(col.mean()) if len(col) > 0 else None
    return breakdown
