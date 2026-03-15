import asyncio
import io
import logging
import time
from typing import Optional, Tuple

from .base import TTSModel

logger = logging.getLogger(__name__)

VOICE = "en-US-JennyNeural"


class EdgeTTSModel(TTSModel):
    """Microsoft Edge TTS wrapper using the edge-tts package."""

    def __init__(self, voice: str = VOICE):
        self._voice = voice

    @property
    def name(self) -> str:
        return "edge-tts"

    def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    async def _generate_async(self, text: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, self._voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        audio_data = buf.read()
        if not audio_data:
            raise RuntimeError(
                "edge-tts returned empty audio. Check network connectivity."
            )
        return audio_data

    def generate(self, text: str, voice_prompt_path: Optional[str] = None) -> Tuple[bytes, float]:
        if not self.is_available():
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        start_time = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._generate_async(text))
                    audio_bytes = future.result()
            else:
                audio_bytes = loop.run_until_complete(self._generate_async(text))
        except RuntimeError:
            audio_bytes = asyncio.run(self._generate_async(text))

        inference_time = time.perf_counter() - start_time
        return audio_bytes, inference_time
