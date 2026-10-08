from src.postprocessing.spellcheck import correct_text, is_spellchecker_available


def test_spellchecker_api_exists():
    # Ensure API callable and returns string
    out = correct_text('hello')
    assert isinstance(out, str)


def test_is_spellchecker_available_returns_bool():
    assert isinstance(is_spellchecker_available(), bool)
