import re
from config import CENSOR_WORDS, TRIM_WHITESPACE
import unicodedata

class PostProcessor:
    def __init__(self, censor_words=None, trim_whitespace=True):
        # Merge default censor words with optional new ones
        self.censor_words = set(word.lower() for word in (censor_words or CENSOR_WORDS))
        self.trim_whitespace = trim_whitespace
        self._compile_regex()

    def _normalize_text(self, text: str) -> str:
        # Normalize Unicode to NFKC and remove Arabic diacritics
        text = unicodedata.normalize('NFKC', text)
        arabic_diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
        return arabic_diacritics.sub('', text)

    def _compile_regex(self):
        if not self.censor_words:
            self.censor_regex = None
            return
        # Create one regex pattern for all censor words (English + Arabic)
        pattern = "|".join(re.escape(word) for word in self.censor_words)
        self.censor_regex = re.compile(pattern, re.IGNORECASE)

    def censor_text(self, text: str) -> str:
        if not self.censor_regex:
            return text
        # Normalize before censoring to catch variants
        normalized_text = self._normalize_text(text)
        return self.censor_regex.sub(lambda m: "*" * len(m.group()), normalized_text)

    def clean_output(self, text: str) -> str:
        text = self.censor_text(text)
        if self.trim_whitespace:
            # Replace multiple spaces with single space
            text = " ".join(text.split())
        return text
