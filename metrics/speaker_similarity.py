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
    from speechbrain.pretrained import EncoderClassifier
    _SPEAKER_MODEL = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    logger.info("Speaker encoder loaded.")
    return _SPEAKER_MODEL


def embed(audio_bytes: bytes) -> np.ndarray:
    import torch
    import torchaudio

    model = _get_speaker_model()
    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()


def compute_similarity(ref_bytes: bytes, gen_bytes: bytes) -> float:
    ref_emb = embed(ref_bytes)
    gen_emb = embed(gen_bytes)

    norm_ref = np.linalg.norm(ref_emb)
    norm_gen = np.linalg.norm(gen_emb)
    if norm_ref == 0 or norm_gen == 0:
        return 0.0
    cos_sim = float(np.dot(ref_emb, gen_emb) / (norm_ref * norm_gen))
    # Map from [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))


class SpeakerSimilarityMetric:
    def __init__(self, reference_audio_bytes: Optional[bytes] = None):
        self.name = "speaker_similarity"
        self._reference_bytes = reference_audio_bytes

    def set_reference(self, audio_bytes: bytes) -> None:
        self._reference_bytes = audio_bytes

    def compute(self, audio_bytes: bytes) -> Dict:
        if self._reference_bytes is None:
            logger.warning("No reference audio set for speaker similarity. Returning 0.")
            return {"speaker_similarity": 0.0}
        sim = compute_similarity(self._reference_bytes, audio_bytes)
        return {"speaker_similarity": sim}
