import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import asyncio
import logging
import regex as re  # better Unicode support for Arabic

# ----------------- Logging -----------------
logger = logging.getLogger("GPT2Engine")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class GPT2Engine:
    def __init__(
        self,
        model_name="gpt2-medium",
        device=None,
        max_length=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        checkpoint_dir="checkpoints",
        fine_tune=False,
        cache_enabled=True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = self._load_bilingual_tokenizer(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.fine_tune = fine_tune
        self.cache_enabled = cache_enabled
        self.response_cache = {}  # in-memory cache

        logger.info(f"Initialized GPT2Engine on device {self.device} with model {model_name}")

    # ----------------- Bilingual Tokenizer -----------------
    def _load_bilingual_tokenizer(self, model_name):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add special handling for Arabic + English mixed input
        # Arabic Unicode block: \u0600-\u06FF
        # Keep Arabic words intact while splitting English normally
        tokenizer._tokenize = lambda text: self._custom_tokenize(text)
        return tokenizer

    def _custom_tokenize(self, text):
        # Split by whitespace and punctuation but keep Arabic words together
        pattern = r"[\p{Arabic}]+|[\w]+|[^\s\w]"
        return self.tokenizer.convert_tokens_to_ids(re.findall(pattern, text))

    # ----------------- Encoding/Decoding -----------------
    def encode(self, text: str):
        return torch.tensor([self.tokenizer.encode(text, add_special_tokens=False)]).to(self.device)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # ----------------- Core Generation -----------------
    def generate_response(self, prompt: str, async_mode=False, **kwargs):
        if self.cache_enabled and prompt in self.response_cache:
            logger.debug("Cache hit for prompt")
            return self.response_cache[prompt]

        if async_mode:
            return asyncio.run(self._generate_async(prompt, **kwargs))
        else:
            response = self._generate_sync(prompt, **kwargs)
            if self.cache_enabled:
                self.response_cache[prompt] = response
            return response

    def _generate_sync(self, prompt: str, **kwargs):
        gen_kwargs = dict(
            max_length=kwargs.get("max_length", self.max_length),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        try:
            input_ids = self.encode(prompt)
            output = self.model.generate(input_ids, **gen_kwargs)
            response = self.decode(output[0][input_ids.shape[-1]:])
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "[!] Error generating response."

    async def _generate_async(self, prompt: str, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, prompt, kwargs)

    # ----------------- Fine-Tuning -----------------
    def fine_tune_on_texts(self, texts, epochs=1, lr=5e-5):
        if not self.fine_tune:
            logger.warning("Fine-tuning disabled.")
            return
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        logger.info(f"Starting fine-tuning for {epochs} epochs on {len(texts)} texts")
        for epoch in range(epochs):
            for i, text in enumerate(texts):
                input_ids = self.encode(text)
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}, text {i+1}/{len(texts)}, loss {loss.item():.4f}")
        self.model.eval()
        logger.info("Fine-tuning completed.")

    # ----------------- Checkpointing -----------------
    def save_checkpoint(self, name="latest.pt"):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info(f"Checkpoint loaded: {path}")
        else:
            logger.warning(f"Checkpoint path does not exist: {path}")

    # ----------------- Cache Management -----------------
    def clear_cache(self):
        self.response_cache.clear()
        logger.debug("Response cache cleared.")
