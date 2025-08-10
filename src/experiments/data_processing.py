from datasets import load_dataset, Dataset
from typing import Dict, Any, List
from utils import get_logger
import json

logger = get_logger(__name__)

# Data augmentation approaches could be added as future improvement of the project


def _process_and_expand_batch(
    batch: Dict[str, List], config: Dict[str, Any]
) -> Dict[str, List]:
    """
    Processes and expands each batch by:
    - For successful entries, expanding each suggestion that meets the confidence threshold.
    - For blocked entries, adding the refusal response.

    Returns a dict with expanded inputs and outputs.
    """
    data_conf = config["data"]
    expanded_inputs, expanded_outputs = [], []
    for i in range(len(batch[data_conf["input_key"]])):
        if batch[data_conf["status_key"]][i] == "success":
            for sug in batch[data_conf["suggestions_key"]][i]:
                if (
                    sug[data_conf["confidence_key"]]
                    >= data_conf["min_confidence_threshold"]
                ):
                    expanded_inputs.append(batch[data_conf["input_key"]][i])
                    expanded_outputs.append(sug[data_conf["domain_key"]])
        elif batch[data_conf["status_key"]][i] == "blocked":
            expanded_inputs.append(batch[data_conf["input_key"]][i])
            expanded_outputs.append(data_conf["refusal_response"])
    return {
        "business_description": expanded_inputs,
        "suggestion_output": expanded_outputs,
    }


def _format_prompt(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    """
    Formats a prompt string for each example using a template from config.
    Inserts the business description and suggestion output into the template.
    """
    return {
        "text": config["data"]["prompt_template"].format(
            business_description=example["business_description"],
            suggestion_output=example["suggestion_output"],
        )
    }


def load_and_prepare_dataset(config: Dict[str, Any]) -> Dataset:
    """
    Loads the dataset from a JSON file, optionally subsets it for debugging,
    processes and expands each example, then formats prompts for training.
    Returns the prepared Dataset ready for training.
    """
    logger.info(f"Loading dataset from {config['data']['dataset_path']}")
    l = config["data"]["dataset_path"]
    dataset = load_dataset("json", data_files=l)["train"]

    # If debug subset size is set, select a smaller subset for faster experimentation
    subset_size = config["data"].get("debug_subset_size")
    if subset_size is not None and subset_size > 0:
        logger.warning(f"!!! USING A DEBUG SUBSET OF {subset_size} EXAMPLES !!!")
        dataset = dataset.shuffle(seed=42).select(range(subset_size))

    # Expand batch data based on confidence thresholds and status, remove original columns
    processed_dataset = dataset.map(
        _process_and_expand_batch,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"config": config},
    )

    # Format each example into a prompt string using the prompt template
    formatted_dataset = processed_dataset.map(
        _format_prompt,
        remove_columns=processed_dataset.column_names,
        fn_kwargs={"config": config},
    )

    logger.info(
        "Dataset prepared for SFTTrainer. Final columns: %s",
        formatted_dataset.column_names,
    )
    return formatted_dataset


def _expand_and_format_batch(
    batch: Dict[str, List], config: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Alternative batch expansion and formatting function:
    - Filters suggestions by confidence threshold.
    - Converts filtered suggestions to JSON string.
    - Formats input description and suggestions into a single text prompt.
    Returns a dict with a single 'text' field (list of prompts).
    """

    data_conf = config["data"]
    inp_key = data_conf["input_key"]
    out_key = data_conf["output_key"]
    threshold = data_conf["min_confidence_threshold"]
    prompt_tmpl = (
        "### Input:\n"
        "{description}\n\n"
        "### Response:\n"
        "{suggestion_json}"
        "<|endoftext|>"
    )

    texts: List[str] = []
    for desc, outputs in zip(batch[inp_key], batch[out_key]):
        filtered = [
            {"domain": s["domain"], "confidence": s["confidence"]}
            for s in outputs
            if s.get("confidence", 0.0) >= threshold
        ]

        json_sug = json.dumps(filtered, ensure_ascii=False)

        text = prompt_tmpl.format(description=desc, suggestion_json=json_sug)
        texts.append(text)

    return {"text": texts}


def load_and_prepare_dataset(config: Dict[str, Any], num_proc: int = 1) -> Dataset:
    """
    Loads a JSONL dataset from path specified in config,
    applies expansion and formatting to each example in one pass,
    and returns a Dataset containing only a 'text' column suitable for training.

    Arguments:
        config: Configuration dict with dataset paths and processing parameters.
        num_proc: Number of processes for parallel mapping.
    """

    path = config["data"]["dataset_path"]
    logger.info("Loading dataset from %s", path)
    ds = load_dataset("json", data_files=path, split="train")

    # debug subset
    dbg = config["data"].get("debug_subset_size", 0)
    if dbg and dbg > 0:
        logger.warning("Using DEBUG subset of %d examples", dbg)
        ds = ds.shuffle(seed=42).select(range(dbg))

    # expand + format in one map call
    ds = ds.map(
        _expand_and_format_batch,
        batched=True,
        remove_columns=ds.column_names,
        fn_kwargs={"config": config},
        num_proc=num_proc,
    )

    logger.info("Prepared dataset with columns: %s", ds.column_names)
    return ds
