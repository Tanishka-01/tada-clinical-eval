import logging
import re
from typing import Dict, List, Optional

from .cer import _normalize_text, transcribe_audio

logger = logging.getLogger(__name__)


def extract_medical_terms(text: str, known_terms: List[str]) -> List[str]:
    text_norm = _normalize_text(text)
    found = []
    for term in known_terms:
        term_norm = _normalize_text(term)
        if term_norm in text_norm:
            found.append(term)
    return found


def compute_cta(
    reference: str,
    hypothesis: str,
    medical_terms: List[str],
) -> Dict:
    if not medical_terms:
        return {
            "cta_score": 1.0,
            "correct_terms": [],
            "incorrect_terms": [],
            "substitutions": {},
            "num_terms": 0,
        }

    hyp_norm = _normalize_text(hypothesis)
    correct = []
    incorrect = []
    substitutions = {}

    for term in medical_terms:
        term_norm = _normalize_text(term)
        if term_norm in hyp_norm:
            correct.append(term)
        else:
            incorrect.append(term)
            # Try to find a partial/phonetic match in hypothesis
            words = term_norm.split()
            found_partial = []
            for word in words:
                if len(word) > 3:
                    pattern = re.compile(r"\b" + re.escape(word[:4]) + r"\w*\b")
                    matches = pattern.findall(hyp_norm)
                    if matches:
                        found_partial.extend(matches)
            substitutions[term] = found_partial[0] if found_partial else "not_found"

    cta_score = len(correct) / len(medical_terms)
    return {
        "cta_score": cta_score,
        "correct_terms": correct,
        "incorrect_terms": incorrect,
        "substitutions": substitutions,
        "num_terms": len(medical_terms),
    }


class CTAMetric:
    def __init__(self):
        self.name = "cta"

    def compute(
        self,
        audio_bytes: bytes,
        reference_text: str,
        medical_terms: List[str],
    ) -> Dict:
        if not medical_terms:
            return {"cta_score": None, "note": "No medical terms for this sentence"}

        hypothesis = transcribe_audio(audio_bytes)
        result = compute_cta(reference_text, hypothesis, medical_terms)
        result["transcription"] = hypothesis
        return result
