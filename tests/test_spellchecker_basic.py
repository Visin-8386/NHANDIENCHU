import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postprocessing.spellcheck import correct_text, is_spellchecker_available


def test_spellchecker_api_exists():
    # Ensure API callable and returns string
    out = correct_text('hello')
    assert isinstance(out, str)


def test_is_spellchecker_available_returns_bool():
    assert isinstance(is_spellchecker_available(), bool)
