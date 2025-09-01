import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import MODEL_NAME, TEMPERATURE, TOP_P, REPETITION_PENALTY

class GPT2Engine:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def generate_response(self, prompt: str, max_length=150):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
