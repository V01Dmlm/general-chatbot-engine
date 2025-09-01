import json
import os
from datetime import datetime
from collections import deque
from textblob import TextBlob
import threading

class ContextManager:
    def __init__(self, max_history=50, persist_file=None, enable_sentiment=True, enable_topic=True):
        self.max_history = max_history
        self.persist_file = persist_file
        self.enable_sentiment = enable_sentiment
        self.enable_topic = enable_topic
        self.lock = threading.Lock()
        self.history = deque(maxlen=max_history)
        if persist_file and os.path.exists(persist_file):
            self.load_history()

    def add_message(self, role, text):
        with self.lock:
            message = {
                "role": role,
                "text": text,
                "timestamp": datetime.utcnow().isoformat()
            }
            if self.enable_sentiment:
                message["sentiment"] = self.analyze_sentiment(text)
            if self.enable_topic:
                message["topic"] = self.extract_topic(text)
            self.history.append(message)
            if self.persist_file:
                self.save_history()

    def analyze_sentiment(self, text):
        try:
            return round(TextBlob(text).sentiment.polarity, 3)
        except:
            return 0.0

    def extract_topic(self, text):
        try:
            blob = TextBlob(text)
            nouns = [word.lemmatize() for word, tag in blob.tags if tag in ("NN", "NNP")]
            return nouns[:5]
        except:
            return []

    def get_history(self):
        with self.lock:
            return list(self.history)

    def clear_history(self):
        with self.lock:
            self.history.clear()
            if self.persist_file:
                self.save_history()

    def save_history(self):
        if not self.persist_file:
            return
        try:
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(list(self.history), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ContextManager] Failed to save: {e}")

    def load_history(self):
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for msg in data:
                    self.history.append(msg)
        except Exception as e:
            print(f"[ContextManager] Failed to load: {e}")

    def trim_history(self, strategy="oldest"):
        with self.lock:
            if len(self.history) <= self.max_history:
                return
            if strategy == "oldest":
                while len(self.history) > self.max_history:
                    self.history.popleft()
            elif strategy == "sentiment":
                sorted_hist = sorted(self.history, key=lambda x: abs(x.get("sentiment", 0)), reverse=True)
                self.history = deque(sorted_hist[:self.max_history], maxlen=self.max_history)

    def get_history_text(self, max_messages=None):
        with self.lock:
            texts = [m["text"] for m in list(self.history)[-max_messages:]]
            return " ".join(texts)
