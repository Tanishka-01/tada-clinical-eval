import logging
import os
import re
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    logger.info("Loading Whisper tiny model (~150MB download on first run)...")
    import whisper
    _WHISPER_MODEL = whisper.load_model("tiny")
    logger.info("Whisper tiny model loaded.")
    return _WHISPER_MODEL


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcribe_audio(audio_bytes: bytes) -> str:
    model = _get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = model.transcribe(tmp_path, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def compute_cer(reference: str, hypothesis: str) -> float:
    from jiwer import cer as jiwer_cer
    ref_norm = _normalize_text(reference)
    hyp_norm = _normalize_text(hypothesis)
    if not ref_norm:
        return 0.0
    return float(jiwer_cer(ref_norm, hyp_norm))


def compute_clinical_cer(
    reference: str,
    hypothesis: str,
    medical_terms: List[str],
) -> Dict:
    overall = compute_cer(reference, hypothesis)
    hyp_norm = _normalize_text(hypothesis)
    correct = []
    errors = []
    for term in medical_terms:
        term_norm = _normalize_text(term)
        if term_norm in hyp_norm:
            correct.append(term)
        else:
            errors.append(term)
    term_accuracy = len(correct) / len(medical_terms) if medical_terms else 1.0
    return {
        "overall_cer": overall,
        "term_accuracy": term_accuracy,
        "terms_errors": errors,
        "terms_correct": correct,
    }


class CERMetric:
    """Convenience wrapper around CER computation."""

    def __init__(self):
        self.name = "cer"

    def compute(
        self,
        audio_bytes: bytes,
        reference_text: str,
        medical_terms: Optional[List[str]] = None,
    ) -> Dict:
        hypothesis = transcribe_audio(audio_bytes)
        overall_cer = compute_cer(reference_text, hypothesis)
        result = {
            "cer": overall_cer,
            "transcription": hypothesis,
        }
        if medical_terms:
            clinical = compute_clinical_cer(reference_text, hypothesis, medical_terms)
            result.update({
                "term_accuracy": clinical["term_accuracy"],
                "terms_errors": clinical["terms_errors"],
            })
        return result
