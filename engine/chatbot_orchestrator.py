from .input_processor import InputProcessor
from .context_manager import ContextManager
from .gpt2_engine import GPT2Engine
from .post_processor import PostProcessor
from config import MAX_HISTORY, LOG_FILE
import json
import os

class ChatbotEngine:
    def __init__(self):
        self.input_processor = InputProcessor()
        self.context_manager = ContextManager(MAX_HISTORY)
        self.gpt2_engine = GPT2Engine()
        self.post_processor = PostProcessor()
        self.log_file = LOG_FILE
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    def log_conversation(self, user_input, bot_response):
        log_entry = {"user": user_input, "bot": bot_response}
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)

    def get_response(self, user_input, conversation_history=None):
        processed = self.input_processor.process(user_input)
        self.context_manager.add_message("user", processed["cleaned_text"])
        
        # Build prompt
        history_text = " ".join([m["text"] for m in self.context_manager.get_history()])
        response = self.gpt2_engine.generate_response(history_text)
        response = self.post_processor.clean_output(response)
        
        self.context_manager.add_message("bot", response)
        self.log_conversation(user_input, response)
        
        return response, self.context_manager.get_history()
