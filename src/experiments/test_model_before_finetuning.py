import torch
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main(model_name: str, description: str):
    logger.info(f"Loading pretrained model: {model_name}")
    device = get_device()
    logger.info(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            torch_dtype=torch.bfloat16 if device == 'mps' else torch.float16,
            device_map="auto" if device != "cpu" else None
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    prompt = (
            "Suggest domain names from the description:\n"
            "Output each example as a single line JSON object (JSON Lines format), with keys:\n"
            '  - "domain": the domain string\n'
            '  - "confidence": a float from 0.00–1.00.\n\n'
            "### Description:\n"
            f"{description}\n\n"
            "### Response:\n"
        )

    logger.info("Generating suggestion...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = response_text.split("### Domain:\n")[-1].strip()

    print("\n" + "="*20 + " INFERENCE RESULT " + "="*20)
    print(f"Description: {description}")
    print(response_text)

    try:
        data = json.loads(generated_part)
        print("Status: Success (valid JSON)")
        print(f"Suggested Domain: {data.get('domain', 'N/A')}")
        print(f"Confidence Score: {data.get('confidence', 'N/A')}")
    except json.JSONDecodeError:
        print("Status: Raw Text")
        print(f"Model's Answer: {generated_part}")

    print("="*64 + "\n")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Test model before finetuning")
    parser.add_argument("--model", type=str, default="google/gemma-2b", help="Hugging Face model name")
    parser.add_argument("--description", type=str, required=True, help="Project description for domain name suggestion")
    args = parser.parse_args()

    main(args.model, args.description)