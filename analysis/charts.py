import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STYLE = {
    "bg": "#1a1a2e",
    "fg": "#e0e0e0",
    "grid": "#333355",
    "colors": ["#7b61ff", "#ff6b6b", "#4ecdc4", "#f7b731"],
    "dpi": 300,
}

NUMERIC_METRICS = ["cer", "rtf", "speaker_similarity", "pas_score", "cta_score", "ecs"]


def generate_all_charts(df: pd.DataFrame, figures_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    models = df["model_name"].unique().tolist()
    colors = {m: STYLE["colors"][i % len(STYLE["colors"])] for i, m in enumerate(models)}

    for func, name in [
        (_radar_chart, "radar"),
        (_metric_comparison_bars, "metric_bars"),
        (_category_heatmap, "heatmap"),
        (_cer_distribution, "cer_dist"),
        (_rtf_comparison, "rtf"),
        (_clinical_hallucination_analysis, "hallucination"),
    ]:
        try:
            path = func(df, models, colors, figures_dir)
            if path:
                generated.append(path)
        except Exception as e:
            logger.warning(f"Chart '{name}' failed: {e}")

    if "ecs" in df.columns and df["ecs"].notna().any():
        try:
            path = _ecs_by_tone(df, models, colors, figures_dir)
            if path:
                generated.append(path)
        except Exception as e:
            logger.warning(f"ECS chart failed: {e}")

    return generated


def _radar_chart(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt

    available = [m for m in NUMERIC_METRICS if m in df.columns]
    if len(available) < 3:
        raise ValueError("Not enough metrics for radar chart")

    metric_means = {}
    for model in models:
        mdf = df[df["model_name"] == model]
        metric_means[model] = {}
        for metric in available:
            col = pd.to_numeric(mdf[metric], errors="coerce").dropna()
            metric_means[model][metric] = float(col.mean()) if len(col) > 0 else 0.0

    all_vals = {m: [metric_means[model].get(m, 0) for model in models] for m in available}
    normed = {}
    for model in models:
        normed[model] = []
        for metric in available:
            vals = all_vals[metric]
            mn, mx = min(vals), max(vals)
            v = metric_means[model][metric]
            if mx == mn:
                normed[model].append(0.5)
            elif metric in ["cer", "rtf"]:
                normed[model].append(1.0 - (v - mn) / (mx - mn))
            else:
                normed[model].append((v - mn) / (mx - mn))

    N = len(available)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    for model in models:
        values = normed[model] + normed[model][:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[model], label=model)
        ax.fill(angles, values, alpha=0.15, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, color=STYLE["fg"], size=10)
    ax.set_yticklabels([])
    ax.set_title(
        "Model Performance Radar\n(All Metrics Normalized 0-1)",
        color=STYLE["fg"], size=13, pad=20,
    )
    ax.tick_params(colors=STYLE["fg"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=STYLE["bg"], labelcolor=STYLE["fg"])

    path = str(figures_dir / "radar_chart.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path


def _metric_comparison_bars(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt

    available = [m for m in NUMERIC_METRICS if m in df.columns]
    if not available:
        raise ValueError("No metrics available")

    cols = 3
    rows = (len(available) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.patch.set_facecolor(STYLE["bg"])

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    x = np.arange(len(models))

    for idx, metric in enumerate(available):
        ax = axes_flat[idx]
        ax.set_facecolor(STYLE["bg"])
        means, cis = [], []
        for model in models:
            col = pd.to_numeric(df[df["model_name"] == model][metric], errors="coerce").dropna()
            if len(col) > 0:
                m = float(col.mean())
                ci = 1.96 * float(col.std()) / np.sqrt(len(col))
                means.append(m)
                cis.append(ci)
            else:
                means.append(0)
                cis.append(0)

        for i, (model, mean, ci) in enumerate(zip(models, means, cis)):
            ax.bar(i, mean, width=0.6, color=colors[model], alpha=0.85, label=model)
            ax.errorbar(i, mean, yerr=ci, fmt="none", color=STYLE["fg"], capsize=5)

        ax.set_title(metric.upper(), color=STYLE["fg"], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, color=STYLE["fg"], fontsize=9)
        ax.tick_params(colors=STYLE["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(color=STYLE["grid"], linestyle="--", alpha=0.5, axis="y")

    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Metric Comparison Across Models (mean ± 95% CI)",
                 color=STYLE["fg"], fontsize=14, y=1.01)
    path = str(figures_dir / "metric_comparison_bars.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path


def _category_heatmap(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    categories = df["category"].unique().tolist() if "category" in df.columns else []
    available = [m for m in ["cer", "rtf", "pas_score", "cta_score"] if m in df.columns]

    rows_data = {}
    for model in models:
        for metric in available:
            row_key = f"{model}\n{metric}"
            row = {}
            for cat in categories:
                sub = df[(df["model_name"] == model) & (df["category"] == cat)]
                col = pd.to_numeric(sub[metric], errors="coerce").dropna()
                row[cat] = float(col.mean()) if len(col) > 0 else np.nan
            rows_data[row_key] = row

    heat_df = pd.DataFrame(rows_data).T

    fig, ax = plt.subplots(
        figsize=(max(8, len(categories) * 2), max(6, len(rows_data) * 0.8))
    )
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    sns.heatmap(
        heat_df.astype(float), ax=ax, cmap="RdYlGn_r",
        annot=True, fmt=".3f", linewidths=0.5,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Metric Scores by Model and Category", color=STYLE["fg"], fontsize=13)
    ax.tick_params(colors=STYLE["fg"])

    path = str(figures_dir / "category_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path


def _cer_distribution(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt

    if "cer" not in df.columns:
        raise ValueError("CER not in df")

    categories = df["category"].unique().tolist() if "category" in df.columns else ["all"]
    fig, axes = plt.subplots(1, len(categories), figsize=(6 * len(categories), 6), sharey=True)
    fig.patch.set_facecolor(STYLE["bg"])
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        ax.set_facecolor(STYLE["bg"])
        data, labels = [], []
        for model in models:
            sub = df[(df["model_name"] == model) & (df["category"] == cat)]
            col = pd.to_numeric(sub["cer"], errors="coerce").dropna()
            if len(col) > 0:
                data.append(col.values)
                labels.append(model)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, model in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(model, "#999"))
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "medians", "fliers"]:
            for item in bp[element]:
                item.set_color(STYLE["fg"])

        ax.axhline(y=0.15, color="#ff6b6b", linestyle="--", linewidth=1.5,
                   label="CER=0.15 threshold")
        ax.set_title(f"CER — {cat.title()}", color=STYLE["fg"], fontsize=11)
        ax.tick_params(colors=STYLE["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(color=STYLE["grid"], linestyle="--", alpha=0.5, axis="y")
        ax.set_ylabel("CER", color=STYLE["fg"])
        ax.legend(facecolor=STYLE["bg"], labelcolor=STYLE["fg"])

    fig.suptitle("CER Distribution by Model and Category", color=STYLE["fg"], fontsize=14)
    path = str(figures_dir / "cer_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path


def _rtf_comparison(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt

    if "rtf" not in df.columns:
        raise ValueError("RTF not in df")

    means, stds = [], []
    for model in models:
        col = pd.to_numeric(df[df["model_name"] == model]["rtf"], errors="coerce").dropna()
        means.append(float(col.mean()) if len(col) > 0 else 0)
        stds.append(float(col.std()) if len(col) > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    x = np.arange(len(models))
    ax.bar(x, means, color=[colors.get(m, "#999") for m in models], alpha=0.85, width=0.5)
    ax.errorbar(x, means, yerr=stds, fmt="none", color=STYLE["fg"], capsize=6)
    ax.axhline(y=1.0, color="#ff6b6b", linestyle="--", linewidth=2, label="RTF = 1.0 (real-time)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models, color=STYLE["fg"])
    ax.set_ylabel("Real-Time Factor (log scale)", color=STYLE["fg"])
    ax.set_title("Mean RTF by Model\n(< 1.0 = faster than real-time)", color=STYLE["fg"])
    ax.tick_params(colors=STYLE["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])
    ax.grid(color=STYLE["grid"], linestyle="--", alpha=0.5, axis="y")
    ax.legend(facecolor=STYLE["bg"], labelcolor=STYLE["fg"])

    path = str(figures_dir / "rtf_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path


def _ecs_by_tone(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt

    tones = df["intended_tone"].unique().tolist() if "intended_tone" in df.columns else []
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    x = np.arange(len(tones))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        means = []
        for tone in tones:
            sub = df[(df["model_name"] == model) & (df["intended_tone"] == tone)]
            col = pd.to_numeric(sub["ecs"], errors="coerce").dropna()
            means.append(float(col.mean()) if len(col) > 0 else 0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9, color=colors.get(model, "#999"),
               label=model, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tones, color=STYLE["fg"])
    ax.set_ylabel("Emotional Consistency Score", color=STYLE["fg"])
    ax.set_title("ECS by Intended Tone and Model", color=STYLE["fg"])
    ax.tick_params(colors=STYLE["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])
    ax.grid(color=STYLE["grid"], linestyle="--", alpha=0.5, axis="y")
    ax.legend(facecolor=STYLE["bg"], labelcolor=STYLE["fg"])

    path = str(figures_dir / "ecs_by_tone.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path


def _clinical_hallucination_analysis(df, models, colors, figures_dir) -> str:
    import matplotlib.pyplot as plt

    available = [m for m in ["cer", "cta_score"] if m in df.columns]
    if not available:
        raise ValueError("Neither CER nor CTA in df")

    clinical_df = df[df["category"] == "clinical"] if "category" in df.columns else df

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    x = np.arange(len(models))
    width = 0.35

    for j, metric in enumerate(available):
        means = []
        for model in models:
            col = pd.to_numeric(
                clinical_df[clinical_df["model_name"] == model][metric],
                errors="coerce"
            ).dropna()
            means.append(float(col.mean()) if len(col) > 0 else 0)
        offset = (j - len(available) / 2 + 0.5) * width
        color = STYLE["colors"][j % len(STYLE["colors"])]
        ax.bar(x + offset, means, width=width * 0.9, color=color,
               label=metric.upper(), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, color=STYLE["fg"])
    ax.set_ylabel("Score", color=STYLE["fg"])
    ax.set_title(
        "Clinical Hallucination Analysis\nOverall CER vs Clinical Terminology Accuracy (CTA)",
        color=STYLE["fg"],
    )
    ax.tick_params(colors=STYLE["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])
    ax.grid(color=STYLE["grid"], linestyle="--", alpha=0.5, axis="y")
    ax.legend(facecolor=STYLE["bg"], labelcolor=STYLE["fg"])

    path = str(figures_dir / "clinical_hallucination_analysis.png")
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path
