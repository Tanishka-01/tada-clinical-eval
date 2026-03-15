import io
import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

THERAPEUTIC_NORMS = {
    "speech_rate": (2.0, 3.5),
    "pitch_std": (15.0, 40.0),
    "pause_ratio": (0.15, 0.35),
}

CLINICAL_NORMS = {
    "speech_rate": (2.5, 4.0),
    "pitch_std": (5.0, 25.0),
    "pause_ratio": (0.10, 0.25),
}

CRISIS_NORMS = {
    "speech_rate": (1.5, 3.0),
    "pitch_std": (10.0, 30.0),
    "pause_ratio": (0.20, 0.40),
}

CATEGORY_NORMS = {
    "therapeutic": THERAPEUTIC_NORMS,
    "clinical": CLINICAL_NORMS,
    "crisis": CRISIS_NORMS,
}


def extract_prosodic_features(audio_bytes: bytes, text: str = "") -> Dict:
    import librosa
    import torchaudio

    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)
    y = waveform.squeeze().numpy()
    if waveform.shape[0] > 1:
        y = waveform.mean(dim=0).numpy()

    if sr != 22050:
        waveform_r = torchaudio.functional.resample(waveform, sr, 22050)
        y = waveform_r.squeeze().numpy()
        sr = 22050

    duration = len(y) / sr
    word_count = len(text.split()) if text else max(1, int(duration * 3))
    speech_rate = word_count / duration if duration > 0 else 0.0

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    except Exception as e:
        logger.warning(f"Pitch extraction failed: {e}")
        pitch_mean = 0.0
        pitch_std = 0.0

    try:
        intervals = librosa.effects.split(y, top_db=30)
        speech_samples = sum(end - start for start, end in intervals)
        total_samples = len(y)
        silence_samples = total_samples - speech_samples
        pause_ratio = silence_samples / total_samples if total_samples > 0 else 0.0
    except Exception as e:
        logger.warning(f"Pause extraction failed: {e}")
        pause_ratio = 0.0

    rms = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    return {
        "speech_rate": speech_rate,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pause_ratio": pause_ratio,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "duration_s": duration,
    }


def score_against_norms(features: Dict, norms: Dict) -> Tuple[float, Dict]:
    in_range = {}
    count = 0
    for feature, (low, high) in norms.items():
        val = features.get(feature, 0.0)
        within = low <= val <= high
        in_range[feature] = {
            "value": val,
            "target_range": (low, high),
            "in_range": within,
        }
        if within:
            count += 1
    pas_score = count / len(norms) if norms else 0.0
    return pas_score, in_range


def compute_pas(audio_bytes: bytes, category: str, text: str = "") -> Dict:
    norms = CATEGORY_NORMS.get(category, CLINICAL_NORMS)
    try:
        features = extract_prosodic_features(audio_bytes, text)
    except Exception as e:
        logger.error(f"Prosodic feature extraction failed: {e}")
        return {
            "pas_score": None,
            "features": {},
            "norms": norms,
            "in_range": {},
            "error": str(e),
        }

    pas_score, in_range = score_against_norms(features, norms)
    return {
        "pas_score": pas_score,
        "features": features,
        "norms": norms,
        "in_range": in_range,
    }


class PASMetric:
    def __init__(self):
        self.name = "pas"

    def compute(self, audio_bytes: bytes, category: str, text: str = "") -> Dict:
        return compute_pas(audio_bytes, category, text)
