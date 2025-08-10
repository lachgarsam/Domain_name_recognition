import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    batch_size: int = 1


def safe_load_model(model_dir: str, device: Optional[str] = None):
    """Loads a Hugging Face model, with a fallback for PEFT adapters."""
    device = device or get_device()
    logger.info(f"Loading model from {model_dir} on device {device}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Loaded HF model directly.")
    except Exception as e:
        logger.info(f"Direct load failed: {e}. Trying PEFT adapter fallback.")
        base_model_path = os.environ.get("FINETUNED_BASE_MODEL")
        if not base_model_path:
            raise RuntimeError("Set FINETUNED_BASE_MODEL env var for PEFT models.")

        base = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        logger.info("Loaded base model and applied PEFT adapter.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class DomainGenerator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or get_device()
        self.model, self.tokenizer = safe_load_model(model_path, device=self.device)
        if self.device != "cpu" and not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)

    def generate_batch(self, prompts: List[str], cfg: GenerationConfig) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), cfg.batch_size):
            batch = prompts[i : i + cfg.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                out_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=cfg.do_sample,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            input_token_len = inputs["input_ids"].shape[1]
            new_tokens = out_tokens[:, input_token_len:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(texts)
        return outputs
