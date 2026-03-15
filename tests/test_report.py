from pathlib import Path

import pandas as pd
import pytest


def _mock_results():
    from evaluation.runner import EvaluationResults, SentenceResult

    results = EvaluationResults(
        models_evaluated=["tada-1b", "edge-tts"],
        metrics_computed=["cer", "rtf", "speaker_similarity", "pas_score"],
        total_sentences=4,
        failed_sentences=[],
        timestamp="2026-03-14T00:00:00",
        duration_seconds=10.0,
    )
    for i in range(2):
        for model in ["tada-1b", "edge-tts"]:
            results.results.append(SentenceResult(
                sentence_id=f"test_{i:03d}",
                model_name=model,
                text=f"Test sentence {i}.",
                category="clinical" if i == 0 else "therapeutic",
                subcategory="test",
                intended_tone="neutral",
                metrics={
                    "cer": 0.05 + i * 0.01,
                    "rtf": 2.5 + i * 0.5,
                    "speaker_similarity": 0.8,
                    "pas_score": 0.6,
                    "cta_score": 0.9,
                    "ecs": None,
                },
                inference_time=1.5,
                success=True,
            ))
    return results


def _mock_stats():
    return {
        "tada-1b": {
            "cer": {"mean": 0.07, "std": 0.02, "n": 2, "ci_95": (0.05, 0.09)},
            "rtf": {"mean": 45.0, "std": 5.0, "n": 2, "ci_95": (40, 50)},
        },
        "edge-tts": {
            "cer": {"mean": 0.04, "std": 0.01, "n": 2, "ci_95": (0.03, 0.05)},
            "rtf": {"mean": 0.8, "std": 0.1, "n": 2, "ci_95": (0.7, 0.9)},
        },
    }


def test_report_generates_pdf(tmp_path):
    from report.generator import generate_report

    output_path = tmp_path / "test_report.pdf"
    generate_report(_mock_results(), _mock_stats(), str(tmp_path / "figures"), str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_report_has_correct_pages(tmp_path):
    from report.generator import generate_report

    output_path = tmp_path / "test_report.pdf"
    generate_report(_mock_results(), _mock_stats(), str(tmp_path / "figures"), str(output_path))

    content = output_path.read_bytes()
    page_count = content.count(b"/Page ")
    assert page_count >= 5, f"Expected at least 5 pages, found ~{page_count}"


def test_tables_generated():
    from analysis.tables import (
        generate_main_results_table,
        generate_category_table,
        generate_cta_analysis_table,
    )

    data = []
    for model in ["tada-1b", "edge-tts"]:
        for cat in ["clinical", "therapeutic"]:
            data.append({
                "model_name": model, "category": cat,
                "cer": 0.05, "rtf": 1.5,
                "speaker_similarity": 0.8, "pas_score": 0.6,
                "cta_score": 0.9, "cta_incorrect_terms": [],
            })
    df = pd.DataFrame(data)

    main = generate_main_results_table(df)
    assert "\\begin{table}" in main

    cat = generate_category_table(df)
    assert len(cat) > 0

    cta = generate_cta_analysis_table(df)
    assert len(cta) > 0


def test_all_figures_created(tmp_path):
    from analysis.charts import generate_all_charts

    data = []
    for model in ["tada-1b", "edge-tts"]:
        for cat in ["clinical", "therapeutic", "crisis"]:
            for tone in ["neutral", "warm", "calm"]:
                data.append({
                    "model_name": model, "category": cat, "intended_tone": tone,
                    "cer": 0.05, "rtf": 1.5, "speaker_similarity": 0.8,
                    "pas_score": 0.6, "cta_score": 0.9, "ecs": 0.7,
                })
    df = pd.DataFrame(data)
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    generated = generate_all_charts(df, str(figures_dir))
    assert len(generated) >= 5
    for fig_path in generated:
        assert Path(fig_path).exists(), f"Missing figure: {fig_path}"
