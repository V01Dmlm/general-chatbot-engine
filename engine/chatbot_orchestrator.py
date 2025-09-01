# engine/chatbot_orchestrator.py
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from config import LOG_FILE, MAX_HISTORY
from .input_processor import InputProcessor
from .context_manager import ContextManager
from .gpt2_engine import GPT2Engine
from .post_processor import PostProcessor
import json
import os
import threading

# ----------------- Logging Setup -----------------
logger = logging.getLogger("ChatbotEngine")
logger.setLevel(logging.DEBUG)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

class ChatbotEngine:
    """
    Overengineered orchestrator for handling input -> GPT2 -> post-processing.
    Features:
        - Bilingual input (English + Arabic)
        - Async response generation
        - Thread-safe caching
        - Logging & conversation persistence
        - Hooks for fine-tuning
        - Redundant safety and error handling
    """

    def __init__(self):
        self.input_processor = InputProcessor()
        self.context_manager = ContextManager(MAX_HISTORY)
        self.gpt2_engine = GPT2Engine()
        self.post_processor = PostProcessor()
        self.lock = threading.Lock()
        self.cache: Dict[str, str] = {}  # in-memory response cache
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    # ----------------- Conversation Logging -----------------
    def log_conversation(self, user_input: str, bot_response: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": bot_response
        }
        logger.debug(f"Logging conversation: {log_entry}")
        with self.lock:
            try:
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                else:
                    logs = []
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []
            logs.append(log_entry)
            try:
                with open(LOG_FILE, "w", encoding="utf-8") as f:
                    json.dump(logs, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"[!] Failed to write log: {e}")

    # ----------------- Async Response -----------------
    async def _generate_response_async(self, prompt: str) -> str:
        if prompt in self.cache:
            logger.debug("Cache hit for prompt")
            return self.cache[prompt]

        def sync_gen():
            try:
                # Automatically detect language and preprocess input
                processed_prompt = self.input_processor.process(prompt)["cleaned"]
                response = self.gpt2_engine.generate_response(processed_prompt)
                return response
            except Exception as e:
                logger.error(f"[!] GPT2 generation failed: {e}")
                return "Sorry, I couldn't generate a response."

        response = await self.loop.run_in_executor(None, sync_gen)
        with self.lock:
            self.cache[prompt] = response
        return response

    # ----------------- Training Hook -----------------
    def fine_tune(self, dataset_path: str, epochs: int = 1, batch_size: int = 2, learning_rate: float = 5e-5):
        import torch
        from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{dataset_path} not found!")

        logger.info(f"Starting fine-tuning on {dataset_path}")
        tokenizer = self.gpt2_engine.tokenizer
        model = self.gpt2_engine.model

        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=dataset_path,
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir="./fine_tuned_gpt2",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=2,
            learning_rate=learning_rate,
            logging_dir="./logs"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        trainer.train()
        trainer.save_model("./fine_tuned_gpt2")
        logger.info("Fine-tuning completed!")

    # ----------------- Get Response -----------------
    def get_response(self, user_input: str, conversation_history: Optional[List[Dict]] = None) -> (str, List[Dict]):
        processed = self.input_processor.process(user_input)
        self.context_manager.add_message("user", processed["cleaned"])

        # Build prompt from conversation history
        history_text = self.context_manager.get_history_text(MAX_HISTORY)
        response = self.loop.run_until_complete(self._generate_response_async(history_text))

        # Post-processing with redundancy checks
        try:
            response = self.post_processor.clean_output(response)
        except Exception as e:
            logger.warning(f"[!] Post-processing failed: {e}")

        self.context_manager.add_message("bot", response)
        self.log_conversation(user_input, response)

        return response, self.context_manager.get_history()

    # ----------------- Export/Import History -----------------
    def export_history(self, file_path: str):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.context_manager.get_history(), f, indent=2, ensure_ascii=False)
            logger.info(f"History exported to {file_path}")
        except Exception as e:
            logger.error(f"[!] Failed to export history: {e}")

    def import_history(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            self.context_manager.history.clear()
            for msg in history:
                self.context_manager.add_message(msg.get("role", "user"), msg.get("text", ""))
            logger.info(f"History imported from {file_path}")
        except Exception as e:
            logger.error(f"[!] Failed to import history: {e}")
        