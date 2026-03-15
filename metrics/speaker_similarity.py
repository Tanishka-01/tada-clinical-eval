import io
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SPEAKER_MODEL = None


def _get_speaker_model():
    global _SPEAKER_MODEL
    if _SPEAKER_MODEL is not None:
        return _SPEAKER_MODEL
    logger.info("Loading resemblyzer VoiceEncoder...")
    from resemblyzer import VoiceEncoder
    _SPEAKER_MODEL = VoiceEncoder(device="cpu")
    logger.info("VoiceEncoder loaded.")
    return _SPEAKER_MODEL


def embed(audio_bytes: bytes) -> Optional[np.ndarray]:
    try:
        import soundfile as sf
        from resemblyzer import preprocess_wav

        model = _get_speaker_model()
        buf = io.BytesIO(audio_bytes)
        wav, sr = sf.read(buf, dtype="float32")
        # preprocess_wav expects mono float32 at 16 kHz
        wav = preprocess_wav(wav, source_sr=sr)
        return model.embed_utterance(wav)
    except Exception as e:
        logger.warning(f"Speaker embedding failed: {e}")
        return None


def compute_similarity(ref_bytes: bytes, gen_bytes: bytes) -> Optional[float]:
    try:
        ref_emb = embed(ref_bytes)
        gen_emb = embed(gen_bytes)

        if ref_emb is None or gen_emb is None:
            return None

        norm_ref = np.linalg.norm(ref_emb)
        norm_gen = np.linalg.norm(gen_emb)
        if norm_ref == 0 or norm_gen == 0:
            return 0.0
        cos_sim = float(np.dot(ref_emb, gen_emb) / (norm_ref * norm_gen))
        return max(0.0, min(1.0, cos_sim))
    except Exception as e:
        logger.warning(f"Speaker similarity computation failed: {e}")
        return None


class SpeakerSimilarityMetric:
    def __init__(self, reference_audio_bytes: Optional[bytes] = None):
        self.name = "speaker_similarity"
        self._reference_bytes = reference_audio_bytes

    def set_reference(self, audio_bytes: bytes) -> None:
        self._reference_bytes = audio_bytes

    def compute(self, audio_bytes: bytes) -> Dict:
        if self._reference_bytes is None:
            logger.warning("No reference audio set for speaker similarity. Returning None.")
            return {"speaker_similarity": None}
        sim = compute_similarity(self._reference_bytes, audio_bytes)
        if sim is None:
            logger.warning("Speaker similarity returned None — will show as N/A in report.")
        return {"speaker_similarity": sim}
