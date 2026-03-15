import pytest


def test_base_model_interface():
    from models.base import TTSModel
    with pytest.raises(TypeError):
        TTSModel()


def test_edge_tts_is_available():
    from models.edge_tts_model import EdgeTTSModel
    model = EdgeTTSModel()
    assert model.is_available(), "edge-tts package not installed — run: pip install edge-tts"


def test_edge_tts_name():
    from models.edge_tts_model import EdgeTTSModel
    assert EdgeTTSModel().name == "edge-tts"


def test_coqui_tts_name():
    from models.coqui_model import CoquiTTSModel
    assert CoquiTTSModel().name == "coqui-tts"


def test_tada_model_name():
    from models.tada_model import TADAModel
    assert TADAModel().name == "tada-1b"


@pytest.mark.slow
def test_edge_tts_generates_audio():
    from models.edge_tts_model import EdgeTTSModel
    model = EdgeTTSModel()
    if not model.is_available():
        pytest.skip("edge-tts not available")
    audio_bytes, inference_time = model.generate("Hello, this is a test.")
    assert len(audio_bytes) > 100
    assert inference_time >= 0.0


@pytest.mark.slow
def test_edge_tts_clinical_sentence():
    from models.edge_tts_model import EdgeTTSModel
    model = EdgeTTSModel()
    if not model.is_available():
        pytest.skip("edge-tts not available")
    audio_bytes, _ = model.generate(
        "Your lithium carbonate dosage has been adjusted to 300 milligrams three times daily."
    )
    assert len(audio_bytes) > 100


@pytest.mark.slow
def test_tada_model_loads_on_cpu():
    from models.tada_model import TADAModel
    model = TADAModel()
    if not model.is_available():
        pytest.skip("TADA not installed")
    assert isinstance(model.is_available(), bool)
