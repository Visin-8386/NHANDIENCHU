import inspect
from src.models.handwriting_model import EncoderDecoderHTR


def test_generate_default_beam_width():
    sig = inspect.signature(EncoderDecoderHTR.generate)
    assert 'beam_width' in sig.parameters
    if sig.parameters['beam_width'].default is not inspect._empty:
        assert int(sig.parameters['beam_width'].default) == 3


def test_beam_search_default_beam_width():
    sig = inspect.signature(EncoderDecoderHTR.beam_search)
    assert 'beam_width' in sig.parameters
    if sig.parameters['beam_width'].default is not inspect._empty:
        assert int(sig.parameters['beam_width'].default) == 3
