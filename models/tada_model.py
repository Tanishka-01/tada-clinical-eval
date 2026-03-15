import io
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import TTSModel

logger = logging.getLogger(__name__)

_TADA_ENCODER = None
_TADA_MODEL = None
_TADA_TOKENIZER = None


def _get_tada_components():
    global _TADA_ENCODER, _TADA_MODEL, _TADA_TOKENIZER
    if _TADA_MODEL is not None:
        return _TADA_ENCODER, _TADA_MODEL, _TADA_TOKENIZER

    logger.warning(
        "Loading TADA-1B on CPU. This may take several minutes and inference "
        "will be slow (30-120 seconds per sentence). This is expected behavior."
    )
    import torch
    from transformers import AutoTokenizer

    try:
        from tada.modules.encoder import Encoder
        from tada.modules.tada import TadaForCausalLM
    except ImportError:
        raise ImportError(
            "TADA model not found. Install with: pip install hume-tada\n"
            "See https://github.com/HumeAI/tada for details."
        )

    # Check HuggingFace authentication — TADA-1B is gated (built on Llama 3.2)
    try:
        from huggingface_hub import whoami
        whoami()
    except Exception:
        raise RuntimeError(
            "HuggingFace authentication required for TADA-1B (gated model).\n"
            "Please login with:\n"
            "    from huggingface_hub import login\n"
            "    login(token='your_hf_token')\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )

    logger.info("Loading TADA encoder from HumeAI/tada-codec ...")
    _TADA_ENCODER = Encoder.from_pretrained(
        "HumeAI/tada-codec", subfolder="encoder"
    ).to("cpu")
    _TADA_ENCODER.eval()

    logger.info("Loading TADA-1B language model ...")
    _TADA_MODEL = TadaForCausalLM.from_pretrained("HumeAI/tada-1b").to("cpu")
    _TADA_MODEL.eval()

    logger.info("Loading tokenizer ...")
    _TADA_TOKENIZER = AutoTokenizer.from_pretrained("HumeAI/tada-1b")

    logger.info("TADA-1B loaded successfully on CPU.")
    return _TADA_ENCODER, _TADA_MODEL, _TADA_TOKENIZER


def _create_silence_wav(duration_seconds: float = 3.0, sample_rate: int = 22050) -> bytes:
    """Create a WAV file of silence as fallback reference audio."""
    import soundfile as sf

    silence = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


class TADAModel(TTSModel):
    """TADA-1B TTS model wrapper for CPU inference."""

    def __init__(self, reference_audio_path: Optional[str] = None):
        self._reference_audio_path = (
            Path(reference_audio_path) if reference_audio_path else None
        )

    @property
    def name(self) -> str:
        return "tada-1b"

    def is_available(self) -> bool:
        try:
            from tada.modules.encoder import Encoder  # noqa: F401
            from tada.modules.tada import TadaForCausalLM  # noqa: F401
            import torch  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_reference_audio(self) -> bytes:
        if self._reference_audio_path and self._reference_audio_path.exists():
            return self._reference_audio_path.read_bytes()
        assets_path = Path(__file__).parent.parent / "assets" / "reference_neutral.wav"
        if assets_path.exists():
            return assets_path.read_bytes()
        logger.warning("No reference audio found. Using 3-second silence as voice prompt.")
        return _create_silence_wav()

    def generate(self, text: str, voice_prompt_path: Optional[str] = None) -> Tuple[bytes, float]:
        import torch
        import torchaudio

        try:
            encoder, model, tokenizer = _get_tada_components()
        except ImportError as e:
            raise RuntimeError(f"TADA not available: {e}")

        ref_path = voice_prompt_path or (
            str(self._reference_audio_path) if self._reference_audio_path else None
        )

        if ref_path and Path(ref_path).exists():
            ref_audio_bytes = Path(ref_path).read_bytes()
        else:
            ref_audio_bytes = self._load_reference_audio()

        try:
            start_time = time.perf_counter()
            with torch.inference_mode():
                inputs = tokenizer(text, return_tensors="pt")

                buf = io.BytesIO(ref_audio_bytes)
                ref_waveform, ref_sr = torchaudio.load(buf)
                if ref_sr != 16000:
                    ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, 16000)
                if ref_waveform.shape[0] > 1:
                    ref_waveform = ref_waveform.mean(dim=0, keepdim=True)

                ref_codes = encoder(ref_waveform)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    voice_prompt=ref_codes,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.8,
                )

                audio_codes = output["audio_codes"]
                waveform = encoder.decode(audio_codes)

            inference_time = time.perf_counter() - start_time

            buf = io.BytesIO()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            torchaudio.save(buf, waveform.cpu(), 22050, format="wav")
            buf.seek(0)
            wav_bytes = buf.read()

            return wav_bytes, inference_time

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CPU memory exhausted generating '{text[:50]}...'. Skipping."
                )
                raise MemoryError(f"OOM during TADA inference: {e}")
            raise
