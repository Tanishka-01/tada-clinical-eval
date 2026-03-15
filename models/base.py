from abc import ABC, abstractmethod
from typing import Optional, Tuple


class TTSModel(ABC):
    """Abstract base class for all TTS model wrappers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name identifier."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, text: str, voice_prompt_path: Optional[str] = None) -> Tuple[bytes, float]:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize.
            voice_prompt_path: Optional path to reference audio for voice cloning.

        Returns:
            Tuple of (wav_bytes, inference_time_seconds)
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model and its dependencies are installed and ready."""
        raise NotImplementedError
