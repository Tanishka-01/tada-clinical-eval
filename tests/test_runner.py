import io
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_wav_bytes(duration_s: float = 1.0) -> bytes:
    sr = 22050
    n = int(sr * duration_s)
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n)) * 32767 * 0.3).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()


def _mock_model(name: str, fail: bool = False):
    model = MagicMock()
    model.name = name
    model.is_available.return_value = True
    if fail:
        model.generate.side_effect = RuntimeError(f"{name} failed")
    else:
        model.generate.return_value = (_make_wav_bytes(), 0.5)
    return model


def _sentences(n: int = 3):
    cats = ["clinical", "therapeutic", "crisis"]
    return [
        {
            "id": f"test_{i:03d}",
            "text": f"This is test sentence number {i}.",
            "category": cats[i % 3],
            "subcategory": "test",
            "contains_medical_terms": False,
            "medical_terms": [],
            "intended_tone": "neutral",
            "expected_speech_rate": "normal",
        }
        for i in range(n)
    ]


_METRIC_PATCHES = {
    "metrics.cer.CERMetric.compute": {"cer": 0.05, "transcription": "test"},
    "metrics.rtf.RTFMetric.compute": {
        "rtf": 0.5, "inference_time_s": 0.5,
        "audio_duration_s": 1.0, "faster_than_realtime": True,
    },
    "metrics.speaker_similarity.SpeakerSimilarityMetric.compute": {"speaker_similarity": 0.85},
    "metrics.pas.PASMetric.compute": {"pas_score": 0.7, "features": {}, "norms": {}, "in_range": {}},
    "metrics.ecs.ECSMetric.compute": {"ecs": None},
}


def test_runner_processes_all_sentences(tmp_path):
    from evaluation.runner import EvaluationRunner

    model = _mock_model("mock")
    runner = EvaluationRunner(output_dir=str(tmp_path))

    with patch("metrics.cer.CERMetric.compute", return_value=_METRIC_PATCHES["metrics.cer.CERMetric.compute"]), \
         patch("metrics.rtf.RTFMetric.compute", return_value=_METRIC_PATCHES["metrics.rtf.RTFMetric.compute"]), \
         patch("metrics.speaker_similarity.SpeakerSimilarityMetric.compute", return_value=_METRIC_PATCHES["metrics.speaker_similarity.SpeakerSimilarityMetric.compute"]), \
         patch("metrics.pas.PASMetric.compute", return_value=_METRIC_PATCHES["metrics.pas.PASMetric.compute"]):
        results = runner.run_full_evaluation(
            sentences=_sentences(3),
            models=[model],
            metrics_config={},
            skip_ecs=True,
        )

    successful = [r for r in results.results if r.success]
    assert len(successful) == 3


def test_runner_skips_failed_sentences(tmp_path):
    from evaluation.runner import EvaluationRunner

    model = _mock_model("fail", fail=True)
    runner = EvaluationRunner(output_dir=str(tmp_path))
    results = runner.run_full_evaluation(
        sentences=_sentences(2),
        models=[model],
        metrics_config={},
        skip_ecs=True,
    )
    assert len(results.failed_sentences) == 2
    for r in results.results:
        assert r.success is False


def test_results_saved_to_json(tmp_path):
    from evaluation.runner import EvaluationResults, SentenceResult
    from evaluation.storage import save_results, load_results

    results = EvaluationResults(
        models_evaluated=["test-model"],
        metrics_computed=["cer"],
        total_sentences=1,
        failed_sentences=[],
        timestamp="2026-03-14T00:00:00",
        duration_seconds=1.0,
    )
    results.results.append(SentenceResult(
        sentence_id="test_001",
        model_name="test-model",
        text="Hello world.",
        category="clinical",
        subcategory="test",
        intended_tone="neutral",
        metrics={"cer": 0.0},
        inference_time=0.5,
        success=True,
    ))

    out = tmp_path / "results.json"
    save_results(results, str(out))
    assert out.exists()

    loaded = load_results(str(out))
    assert len(loaded.results) == 1
    assert loaded.results[0].sentence_id == "test_001"
    assert loaded.results[0].metrics["cer"] == 0.0


def test_runner_saves_intermediate_results(tmp_path):
    from evaluation.runner import EvaluationRunner

    model = _mock_model("mock")
    runner = EvaluationRunner(output_dir=str(tmp_path))

    with patch("metrics.cer.CERMetric.compute", return_value=_METRIC_PATCHES["metrics.cer.CERMetric.compute"]), \
         patch("metrics.rtf.RTFMetric.compute", return_value=_METRIC_PATCHES["metrics.rtf.RTFMetric.compute"]), \
         patch("metrics.speaker_similarity.SpeakerSimilarityMetric.compute", return_value=_METRIC_PATCHES["metrics.speaker_similarity.SpeakerSimilarityMetric.compute"]), \
         patch("metrics.pas.PASMetric.compute", return_value=_METRIC_PATCHES["metrics.pas.PASMetric.compute"]):
        runner.run_full_evaluation(
            sentences=_sentences(12),
            models=[model],
            metrics_config={},
            skip_ecs=True,
        )

    intermediate = tmp_path / "intermediate_results.json"
    assert intermediate.exists()
    with open(intermediate) as f:
        data = json.load(f)
    assert "results" in data
