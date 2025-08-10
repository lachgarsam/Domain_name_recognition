import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Initialize and return a logger instance with INFO level and a standard format.

    Parameters:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def get_device() -> str:
    """
    Detect the best available device for PyTorch computations.

    Checks for CUDA-enabled GPUs, Apple Silicon MPS backend, and defaults to CPU.

    Returns:
        str: The device string identifier ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        logger.info("NVIDIA CUDA device detected. Using 'cuda' backend.")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("Apple Silicon device detected. Using 'mps' backend.")
        return "mps"
    else:
        logger.info("No specialized hardware detected. Falling back to 'cpu'.")
        return "cpu"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file with support for 'defaults' base config merging.

    If a 'defaults' key exists, it loads the base config and merges with current config,
    overriding or updating nested keys as needed.

    Parameters:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: The merged configuration dictionary.
    """
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "defaults" in config:
        base_path = config_path.parent
        base_config_path = base_path / config["defaults"][0]
        base_config = load_config(str(base_config_path))
        for key, value in config.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value
        return base_config
    return config


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary by concatenating nested keys into a single level dict.

    Parameters:
        d (Dict[str, Any]): The dictionary to flatten.
        parent_key (str, optional): The base key string for recursion (default is '').
        sep (str, optional): Separator to use between concatenated keys (default is '.').

    Returns:
        Dict[str, Any]: A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
