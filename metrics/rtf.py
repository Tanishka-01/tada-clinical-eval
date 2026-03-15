import io
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def get_audio_duration(audio_bytes: bytes) -> float:
    import torchaudio
    buf = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buf)
    return float(waveform.shape[-1] / sample_rate)


def compute_rtf(audio_bytes: bytes, inference_time: float) -> float:
    duration = get_audio_duration(audio_bytes)
    if duration <= 0:
        logger.warning("Audio duration is zero or negative, returning RTF=inf")
        return float("inf")
    return inference_time / duration


class RTFMetric:
    def __init__(self):
        self.name = "rtf"

    def compute(self, audio_bytes: bytes, inference_time: float) -> Dict:
        duration = get_audio_duration(audio_bytes)
        rtf = inference_time / duration if duration > 0 else float("inf")
        return {
            "rtf": rtf,
            "inference_time_s": inference_time,
            "audio_duration_s": duration,
            "faster_than_realtime": rtf < 1.0,
        }
