class PostProcessor:
    def clean_output(self, text: str) -> str:
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
