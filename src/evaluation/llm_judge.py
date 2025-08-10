import json
import re
from typing import List, Dict, Any, Optional
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMDomainJudge:
    """
    A class to evaluate the overall quality of domain name suggestions for a given business description
    by querying a large language model (LLM) through a text generation pipeline.
    """

    def __init__(
        self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "mps"
    ):
        """
        Initialize the judge with a specified language model.

        Args:
            model_id (str): The identifier of the pretrained model to load.
            device (str): The device on which to run the model (e.g., "cpu", "cuda", "mps").
        """
        logger.info(f"Loading judge model: {model_id}")
        self.pipe = pipeline(
            "text-generation", model=model_id, device=device, max_new_tokens=1000
        )

        self.quality_judge_template = (
            "<start_of_turn>user\n"
            "You are a domain naming expert. I will provide a business description and a list of domain suggestions. "
            "Your task is to provide one single, overall rating for the entire list.\n\n"
            "**Business Description:**\n{description}\n\n"
            "**Domain Suggestions:**\n{suggestions}\n\n"
            "**Your Task & Rules:**\n"
            "1. Evaluate the suggestions based on their relevance, brandability, and creativity.\n"
            "2. Respond with ONLY ONE single JSON object that represents the overall quality of the entire list.\n"
            '3. The JSON object must be in the format {{"rating": <1-5>, "reason": "<brief explanation>"}}.\n'
            "4. IMPORTANT: Do not output multiple JSON objects. Just one for the overall list.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the first valid JSON object from a given text string.

        Args:
            text (str): The text potentially containing a JSON object.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON dictionary if found and valid, else None.
        """
        # Use a regex to find a string that looks like a JSON object
        text = text if text[-1] == "}" else text + """.  UNFINISHED SENTENCE"}"""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to decode JSON from extracted string: {json_str}\nError: {e}"
                )
                return None
        logger.warning(f"No JSON object found in the model's output: {text}")
        return None

    def rate_predictions(
        self,
        description: str,
        domain_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use the LLM to evaluate the overall quality of domain suggestions.

        Args:
            description (str): Business description to be used as context.
            domain_outputs (List[Dict[str, Any]]): List of dictionaries containing domain suggestions, expected
                                                   to have a 'domain' key.

        Returns:
            Dict[str, Any]: A dictionary containing the rating (int 1-5) and a brief reason string.
        """
        suggestions_str = "\n".join([f"- {d['domain']}" for d in domain_outputs])
        prompt = self.quality_judge_template.format(
            description=description, suggestions=suggestions_str
        )

        logger.info("Sending prompt to judge...")

        try:
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=75, 
                return_full_text=False,
                num_return_sequences=1,
                eos_token_id=self.pipe.tokenizer.eos_token_id,
            )
            raw_output = outputs[0]["generated_text"].strip()
            logger.info(f"Model raw output:\n{raw_output}")

            # Extract JSON from the potentially messy output
            parsed_json = self._extract_json(raw_output)

            if parsed_json and "rating" in parsed_json:
                # Ensure rating is an integer between 1 and 5
                parsed_json["rating"] = max(
                    1, min(5, int(parsed_json.get("rating", 0)))
                )
                return parsed_json
            else:
                raise ValueError("Parsed JSON is invalid or missing 'rating' key.")

        except Exception as e:
            logger.warning(
                f"Could not parse judge response: {e}\nFalling back to default."
            )
            return {
                "rating": 0,
                "reason": "Could not parse rating from model, defaulting to 0.",
            }
