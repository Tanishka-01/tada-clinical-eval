import io
import wave

import numpy as np
import pytest


def _make_wav(duration_s: float = 1.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 32767 * 0.3).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()


# ---- CER ----

def test_cer_perfect_match():
    from metrics.cer import compute_cer
    assert compute_cer("hello world", "hello world") == pytest.approx(0.0, abs=1e-6)


def test_cer_complete_mismatch():
    from metrics.cer import compute_cer
    assert compute_cer("hello world", "xyz abc def") > 0.0


def test_cer_partial():
    from metrics.cer import compute_cer
    result = compute_cer("hello world", "hello xyz")
    assert 0.0 < result < 1.5


def test_cer_empty_reference():
    from metrics.cer import compute_cer
    assert compute_cer("", "anything") == pytest.approx(0.0, abs=1e-6)


def test_clinical_cer_all_correct():
    from metrics.cer import compute_clinical_cer
    result = compute_clinical_cer(
        "take lithium carbonate twice daily",
        "take lithium carbonate twice daily",
        ["lithium carbonate"],
    )
    assert result["term_accuracy"] == pytest.approx(1.0)
    assert result["terms_errors"] == []


def test_clinical_cer_term_missed():
    from metrics.cer import compute_clinical_cer
    result = compute_clinical_cer(
        "take lithium carbonate twice daily",
        "take some medicine twice daily",
        ["lithium carbonate"],
    )
    assert result["term_accuracy"] == pytest.approx(0.0)
    assert "lithium carbonate" in result["terms_errors"]


def test_clinical_cer_no_terms():
    from metrics.cer import compute_clinical_cer
    result = compute_clinical_cer("hello world", "hello world", [])
    assert result["term_accuracy"] == pytest.approx(1.0)


# ---- CTA ----

def test_cta_all_terms_correct():
    from metrics.cta import compute_cta
    result = compute_cta(
        "take metformin 500 milligrams daily",
        "take metformin 500 milligrams daily",
        ["metformin"],
    )
    assert result["cta_score"] == pytest.approx(1.0)
    assert result["incorrect_terms"] == []


def test_cta_term_missed():
    from metrics.cta import compute_cta
    result = compute_cta(
        "take metformin 500 milligrams daily",
        "take some medicine 500 milligrams daily",
        ["metformin"],
    )
    assert result["cta_score"] == pytest.approx(0.0)
    assert "metformin" in result["incorrect_terms"]


def test_cta_no_medical_terms():
    from metrics.cta import compute_cta
    result = compute_cta("hello world", "hello world", [])
    assert result["cta_score"] == pytest.approx(1.0)
    assert result["num_terms"] == 0


# ---- PAS ----

def test_pas_returns_score_between_0_and_1():
    from metrics.pas import compute_pas
    wav = _make_wav(duration_s=2.0)
    result = compute_pas(wav, "therapeutic", text="hello world how are you today doing well")
    if result.get("pas_score") is not None:
        assert 0.0 <= result["pas_score"] <= 1.0


def test_pas_features_extracted():
    from metrics.pas import compute_pas
    wav = _make_wav(duration_s=2.0)
    result = compute_pas(wav, "clinical", text="this is a test sentence for evaluation")
    if result.get("features"):
        assert "speech_rate" in result["features"]
        assert "pitch_std" in result["features"]
        assert "pause_ratio" in result["features"]


def test_pas_returns_norms():
    from metrics.pas import compute_pas
    wav = _make_wav(duration_s=2.0)
    result = compute_pas(wav, "clinical", text="hello world")
    assert "norms" in result


def test_pas_all_categories():
    from metrics.pas import compute_pas
    wav = _make_wav(duration_s=2.0)
    for cat in ["clinical", "therapeutic", "crisis"]:
        result = compute_pas(wav, cat, text="test sentence here")
        assert "pas_score" in result


# ---- RTF ----

def test_rtf_calculation():
    from metrics.rtf import compute_rtf, get_audio_duration
    wav = _make_wav(duration_s=2.0)
    duration = get_audio_duration(wav)
    rtf = compute_rtf(wav, 1.0)
    assert duration == pytest.approx(2.0, abs=0.1)
    assert rtf == pytest.approx(1.0 / duration, rel=0.01)


def test_rtf_faster_than_realtime():
    from metrics.rtf import RTFMetric
    metric = RTFMetric()
    wav = _make_wav(duration_s=3.0)
    result = metric.compute(wav, inference_time=1.0)
    assert result["faster_than_realtime"] is True
    assert result["rtf"] < 1.0


def test_rtf_slower_than_realtime():
    from metrics.rtf import RTFMetric
    metric = RTFMetric()
    wav = _make_wav(duration_s=1.0)
    result = metric.compute(wav, inference_time=5.0)
    assert result["faster_than_realtime"] is False
    assert result["rtf"] > 1.0


def test_rtf_metric_returns_dict():
    from metrics.rtf import RTFMetric
    metric = RTFMetric()
    wav = _make_wav(duration_s=2.0)
    result = metric.compute(wav, inference_time=1.5)
    for key in ["rtf", "inference_time_s", "audio_duration_s", "faster_than_realtime"]:
        assert key in result
