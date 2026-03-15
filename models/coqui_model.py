import io
import logging
import warnings
from typing import Optional, Tuple
import time

import numpy as np

from .base import TTSModel

logger = logging.getLogger(__name__)

MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

_COQUI_TTS = None


def _get_coqui_tts():
    global _COQUI_TTS
    if _COQUI_TTS is not None:
        return _COQUI_TTS

    logger.info(f"Loading Coqui TTS model: {MODEL_NAME}")
    logger.info("First run will download the model (~100MB). Please wait...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from TTS.api import TTS
        _COQUI_TTS = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)
    logger.info("Coqui TTS model loaded successfully.")
    return _COQUI_TTS


class CoquiTTSModel(TTSModel):
    """Coqui TTS wrapper using the TTS package (CPU compatible)."""

    @property
    def name(self) -> str:
        return "coqui-tts"

    def is_available(self) -> bool:
        try:
            from TTS.api import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, text: str, voice_prompt_path: Optional[str] = None) -> Tuple[bytes, float]:
        if not self.is_available():
            raise ImportError("Coqui TTS not installed. Run: pip install TTS")

        import soundfile as sf

        tts = _get_coqui_tts()

        start_time = time.perf_counter()
        wav_list = tts.tts(text=text)
        inference_time = time.perf_counter() - start_time

        wav_array = np.array(wav_list, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, wav_array, samplerate=22050, format="WAV")
        buf.seek(0)
        wav_bytes = buf.read()

        return wav_bytes, inference_time
