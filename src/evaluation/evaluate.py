import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
from statistics import mean
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llm_judge import LLMDomainJudge
from experiments.utils import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class FineTunedSuggester:
    """
    Wrapper for a fine-tuned PEFT causal language model that generates domain suggestions.

    The model is assumed to output JSON arrays of objects in the format:
      [{"domain": "...", "confidence": 0.87}, ...]
    """

    def __init__(self, model_path: str, device: str) -> None:
        """
        Load the fine-tuned base and PEFT adapter model, along with its tokenizer.

        Args:
            model_path (str): Path or model ID of the fine-tuned model.
            device (str): Device identifier (e.g., 'cpu', 'cuda', 'mps').
        """
        # load base + adapter
        logger.info(f"Loading fine-tuned model from {model_path}")
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float16,
        )
        self.model = PeftModel.from_pretrained(base, model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # self.tokenizer.pad_token = self.tokenizer.eos_token_id
        self.device = device

    def suggest(
        self, description: str, max_new_tokens: int = 250
    ) -> List[Dict[str, Any]]:
        """
        Generate domain suggestions based on a business description.

        Args:
            description (str): Business description prompt.
            max_new_tokens (int): Maximum tokens to generate.

        Returns:
            List[Dict[str, Any]]: List of suggestion dictionaries with 'domain' and 'confidence' keys.
        """

        prompt = (
            "Suggest domain names from the description:\n"
            "Output each example as a single line JSON object (JSON Lines format), with keys:\n"
            '  - "domain": the domain string\n'
            '  - "confidence": a float from 0.00–1.00.\n\n'
            "### Description:\n"
            f"{description}\n\n"
            "### Response:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # extract JSON array
        try:
            start, end = text.find("["), text.rfind("]") + 1
            arr = json.loads(text[start:end])
            return [
                {"domain": e["domain"], "confidence": float(e["confidence"])}
                for e in arr
            ]
        except Exception:
            # fallback parsing lines
            lines = [l.strip() for l in text.splitlines() if l.strip().startswith("-")]
            suggestions = []
            for line in lines[:3]:
                # e.g. "- FooBar.com (0.85)"
                parts = line.replace(",", "").split()
                dom = parts[1]
                try:
                    conf = float(parts[-1].strip("()"))
                except:
                    conf = 1.0
                suggestions.append({"domain": dom, "confidence": conf})
            return suggestions


def evaluate_batch(
    descriptions: List[str],
    suggester: FineTunedSuggester,
    judge: LLMDomainJudge,
) -> List[Dict[str, Any]]:
    """
    Generate domain suggestions and evaluate them for a batch of descriptions.

    Args:
        descriptions (List[str]): List of business descriptions.
        suggester (FineTunedSuggester): Instance of FineTunedSuggester to generate suggestions.
        judge (LLMDomainJudge): Instance of LLMDomainJudge to rate suggestions.

    Returns:
        List[Dict[str, Any]]: List of evaluation records for each description.
    """
    records = []
    for desc in descriptions:
        logger.info(f"Processing description: {desc}")
        sug = suggester.suggest(desc)
        eval_ = judge.rate_predictions(desc, sug)
        records.append({"description": desc, "suggestions": sug, "evaluation": eval_})
    return records


def main(
    descriptions_path: str,
    model_path: str,
    judge_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> None:
    """
    Main driver function to load descriptions, generate suggestions, and evaluate them.

    Args:
        descriptions_path (str): Path to JSONL or JSON file containing descriptions.
        model_path (str): Path to fine-tuned domain suggestion model.
        judge_model (str): Model ID for the LLM judge.
    """

    device = get_device()
    # load suggester & judge
    suggester = FineTunedSuggester(model_path, device)
    judge = LLMDomainJudge(model_id=judge_model)
    with open(descriptions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        descs = [entry["input"] for entry in data]
    # run evaluation
    results = evaluate_batch(descs, suggester, judge)
    # compute average score (skip None)
    scores = [
        r["evaluation"]["rating"]
        for r in results
        if isinstance(r["evaluation"].get("rating"), int)
    ]
    avg = mean(scores) if scores else None

    # output
    out = {"average_rating": avg, "per_example": results}
    print(json.dumps(out, indent=2))

    print("average_rating", out["average_rating"])


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True, help="Path to your fine-tuned model folder"
    )
    parser.add_argument(
        "--judge_model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Judge LLM model ID",
    )
    parser.add_argument(
        "--descriptions_path",
        required=True,
        help='Path to a JSONL file with {"input": str} per line',
    )
    args = parser.parse_args()

    main(args.descriptions_path, args.model_path, args.judge_model)
