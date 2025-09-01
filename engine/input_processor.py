import re
import string
import emoji
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from textblob import TextBlob
from langdetect import detect, DetectorFactory
import nltk

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class InputProcessor:
    def __init__(
        self,
        enable_spellcheck=True,
        enable_lemmatization=True,
        enable_sentiment=True,
        profanity_list=None,
        remove_punctuation=True,
        lowercase=True
    ):
        self.spell = SpellChecker() if enable_spellcheck else None
        self.lemmatizer = WordNetLemmatizer() if enable_lemmatization else None
        self.profanity_list = profanity_list or ["badword1", "badword2"]
        self.enable_sentiment = enable_sentiment
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.arabic_tokenizer = RegexpTokenizer(r'\w+|[^\w\s]', flags=re.UNICODE)

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return lang
        except:
            return "en"  # default to English

    def clean_text(self, text: str, lang="en") -> str:
        if self.lowercase and lang != "ar":  # don't lowercase Arabic
            text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = re.sub(r"\s+", " ", text).strip()

        if self.remove_punctuation:
            if lang == "ar":
                # Keep Arabic letters, remove most punctuation
                text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
            else:
                text = text.translate(str.maketrans("", "", string.punctuation))

        for word in self.profanity_list:
            text = re.sub(rf"\b{word}\b", "*" * len(word), text, flags=re.IGNORECASE)

        return text

    def spell_correction(self, text: str, lang="en") -> str:
        if not self.spell or lang != "en":
            return text
        return " ".join([self.spell.correction(t) or t for t in text.split()])

    def lemmatize_text(self, text: str, lang="en") -> str:
        if not self.lemmatizer or lang != "en":
            return text
        tokens = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(t) for t in tokens])

    def tokenize(self, text: str, lang="en"):
        if lang == "ar":
            return self.arabic_tokenizer.tokenize(text)
        else:
            return word_tokenize(text)

    def sentiment_score(self, text: str, lang="en") -> float:
        if not self.enable_sentiment:
            return 0.0
        try:
            if lang == "en":
                return round(TextBlob(text).sentiment.polarity, 3)
            else:
                return 0.0  # fallback for Arabic
        except:
            return 0.0

    def process(self, text: str) -> dict:
        lang = self.detect_language(text)
        cleaned = self.clean_text(text, lang)
        corrected = self.spell_correction(cleaned, lang)
        lemmatized = self.lemmatize_text(corrected, lang)
        tokens = self.tokenize(lemmatized, lang)
        sentiment = self.sentiment_score(lemmatized, lang)

        return {
            "original": text,
            "language": lang,
            "cleaned": cleaned,
            "corrected": corrected,
            "lemmatized": lemmatized,
            "tokens": tokens,
            "sentiment": sentiment
        }
