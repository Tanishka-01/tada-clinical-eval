"""Generate assets/reference_neutral.wav — 3 seconds of silence at 22050 Hz."""

from pathlib import Path

import numpy as np
import soundfile as sf


def create_reference_audio() -> None:
    out_path = Path(__file__).parent / "reference_neutral.wav"
    silence = np.zeros(int(22050 * 3), dtype=np.float32)
    sf.write(str(out_path), silence, 22050)
    print(f"Created {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_reference_audio()
