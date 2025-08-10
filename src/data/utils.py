import re
import json
import torch
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import OUTPUT_FILE, BLACKLIST


def sanitize(name):
    """
    Sanitize a domain name candidate by removing any triple consecutive characters.

    Parameters:
        name (str): The raw domain name (without TLD).

    Returns:
        str: A cleaned version of the name.
    """
    name = re.sub(r"(.)\1{2,}", r"\1", name)  # remove triple letters
    return name


def is_safe(domain: str) -> bool:
    """
    Check whether a domain name is safe and does not contain blacklisted words.

    Parameters:
        domain (str): The domain name to evaluate.

    Returns:
        bool: True if the domain is considered safe, False otherwise.
    """
    return not any(bad in domain.lower() for bad in BLACKLIST)


def generate_business_description(industry: str, keyword: str) -> str:
    """
    Generate a simple, synthetic business description using the industry and a keyword.

    Parameters:
        industry (str): The industry category (e.g., 'Tech Startup').
        keyword (str): A keyword representing a business concept.

    Returns:
        str: A business description sentence.
    """
    templates = [
        f"A {industry.lower()} focusing on {keyword}.",
        f"A startup in the {industry.lower()} space leveraging {keyword}.",
        f"A {industry.lower()} brand built around {keyword} and innovation.",
        f"An emerging {industry.lower()} service specializing in {keyword}.",
    ]
    return random.choice(templates)


def save_dataset(dataset, filename=OUTPUT_FILE):
    """
    Save the dataset to a JSON file with indentation for readability.

    Parameters:
        dataset (list): The dataset generated from `generate_dataset`.
        filename (str): The filename to save to (default is OUTPUT_FILE).
    """
    Path(filename).write_text(json.dumps(dataset, indent=2))


def load_model(model_name: str, device: str):
    """
    Load a Hugging Face text-generation pipeline with specified model and device.

    Parameters:
        model_name (str): The Hugging Face model identifier.
        device (int): Device ordinal (e.g., 0 for first GPU, -1 for CPU).

    Returns:
        transformers.Pipeline: The loaded text-generation pipeline.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, torch_dtype="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def get_device() -> str:
    """
    Automatically select the best available device for PyTorch.

    Checks availability in the following order:
    1. CUDA GPU
    2. Apple Silicon MPS (Metal Performance Shaders)
    3. CPU fallback

    Returns:
        str: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
