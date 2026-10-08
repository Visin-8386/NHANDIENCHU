import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postprocessing.spellcheck import SpellCorrector

def test_basic_correction():
    sc = SpellCorrector(language='en')
    assert sc.correct_text('helo') == 'hello'

def test_no_change():
    sc = SpellCorrector(language='en')
    assert sc.correct_text('hello') == 'hello'

def test_numeric():
    sc = SpellCorrector(language='en')
    assert sc.correct_text('h3llo') == 'h3llo'  # numbers typically left intact
