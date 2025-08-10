import re
import json
import torch

from typing import Optional, Any


def get_device() -> str:
    """Gets the best available hardware device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_json_array(raw: str) -> Optional[Any]:
    """
    Tries to extract a JSON array or object from a model-generated string.
    """
    if raw is None:
        return None
    s = raw.strip()
    s = re.sub(r"^<Response>\s*", "", s)
    s = re.sub(r"<\|endoftext\|>$", "", s)
    s = re.sub(r"\|+$", "", s).strip()

    def find_matching_bracket(text, open_ch, close_ch, start_idx):
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    # Try to find a JSON array first, then a JSON object
    for brackets in [("[", "]"), ("{", "}")]:
        open_b, close_b = brackets
        start_idx = s.find(open_b)
        if start_idx != -1:
            end_idx = find_matching_bracket(s, open_b, close_b, start_idx)
            if end_idx != -1:
                candidate = s[start_idx : end_idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    return None
