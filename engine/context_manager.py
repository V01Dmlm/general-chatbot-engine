class ContextManager:
    def __init__(self, max_history=3):
        self.max_history = max_history
        self.history = []

    def add_message(self, role: str, text: str):
        self.history.append({"role": role, "text": text})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history(self):
        return self.history
