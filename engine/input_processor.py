import re

class InputProcessor:
    def __init__(self):
        self.greeting_keywords = ["hi", "hello", "hey", "greetings"]

    def clean_input(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\?\!\.,]", "", text)
        return text

    def detect_intent(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in self.greeting_keywords):
            return "greeting"
        elif text_lower.endswith("?"):
            return "question"
        else:
            return "statement"

    def process(self, text: str) -> dict:
        cleaned = self.clean_input(text)
        intent = self.detect_intent(cleaned)
        return {"cleaned_text": cleaned, "intent": intent}
