import sys
sys.path.append('D:/WEB_AI')
from src.postprocessing.spellcheck import correct_text, is_spellchecker_available

print('SPELLCHECKER_AVAILABLE=', is_spellchecker_available())
print('CORRECTION:', correct_text('helo'))
