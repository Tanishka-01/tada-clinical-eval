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
    logger.info("Loading SpeechBrain speaker encoder (downloads on first run)...")
    try:
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        # Fallback for older speechbrain versions
        from speechbrain.pretrained import EncoderClassifier
    _SPEAKER_MODEL = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    logger.info("Speaker encoder loaded.")
    return _SPEAKER_MODEL


def embed(audio_bytes: bytes) -> Optional[np.ndarray]:
    import torch
    import torchaudio

    try:
        model = _get_speaker_model()
    except Exception as e:
        logger.warning(f"Speaker model failed to load: {e}")
        return None

    try:
        buf = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buf)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        with torch.no_grad():
            embedding = model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()
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
        # Map from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
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
