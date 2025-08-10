import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from llm_judge import LLMDomainJudge
from experiments.utils import get_device


def test_llm_judge():
    """
    Test the LLMDomainJudge class with a realistic and a poor domain suggestion list.

    This function:
    - Creates a business description
    - Creates two sets of domain suggestions (good and poor)
    - Runs the judge model on both sets
    - Prints the JSON-formatted results
    """
    description = "An online marketplace for buying and selling handmade skincare."
    # Using more realistic, brandable names for a better test case
    predictions = [
        {"domain": "artisanskin.com", "confidence": 0.92},
        {"domain": "purelycrafted.co", "confidence": 0.88},
        {"domain": "theglowfoundry.market", "confidence": 0.83},
    ]

    judge = LLMDomainJudge(
        model_id="mistralai/Mistral-7B-Instruct-v0.2", device=get_device()
    )
    result = judge.rate_predictions(description, predictions)

    print("\n === Judge Result === ")
    print(json.dumps(result, indent=2))

    # Example of poor domains
    print("\n--- Testing with poor domains ---")
    poor_predictions = [
        {"domain": "cofee.com", "confidence": 0.92},
        {"domain": "salad.co", "confidence": 0.88},
        {"domain": "Ndw.mat", "confidence": 0.83},
    ]
    poor_result = judge.rate_predictions(description, poor_predictions)
    print("\n === Judge Result (Poor) === ")
    print(json.dumps(poor_result, indent=2))


if __name__ == "__main__":

    test_llm_judge()
