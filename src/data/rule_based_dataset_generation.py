import re
import random
from tqdm import tqdm

from config import NUM_SAMPLES, INDUSTRIES, DOMAINS_PER_SAMPLE, OUTPUT_FILE
from utils import is_safe, generate_business_description, save_dataset
from rule_based_generation_strategies import STRATEGIES


def score_domain(domain: str, keyword: str) -> float:
    """
    Generate a pseudo-confidence score (between 0.0 and 1.0) that reflects how
    well the domain matches the given keyword. This is a heuristic estimate.

    Parameters:
        domain (str): The domain name to evaluate.
        keyword (str): The keyword related to the business.

    Returns:
        float: A confidence score in [0.0, 1.0].
    """
    score = 0.5

    domain_l = domain.lower()

    # Keyword relevance
    if keyword.lower() in domain_l:
        score += 0.2

    # Pronounceable
    if re.match(r"^[a-zA-Z]{4,}$", domain.split(".")[0]):
        score += 0.2

    # Good length (7–15 characters before TLD)
    name_part = domain.split(".")[0]
    if 7 <= len(name_part) <= 15:
        score += 0.1

    # Penalize numbers in the middle
    if re.search(r"[a-zA-Z]+\d+[a-zA-Z]+", name_part):
        score -= 0.1

    return round(min(max(score, 0.0), 1.0), 2)


def generate_dataset(num_samples=NUM_SAMPLES):
    """
    Generate a synthetic dataset of business descriptions and domain name suggestions
    with confidence scores.

    Parameters:
        num_samples (int): Number of dataset samples to generate.
        domains_per_sample (int): Number of domain suggestions per sample.

    Returns:
        list[dict]: A list of dataset entries in the format:
            {
                "input": <business_description>,
                "output": [
                    {"domain": <domain_name>, "confidence": <float>},
                    ...
                ]
            }
    """
    dataset = []

    for _ in tqdm(range(num_samples)):
        industry = random.choice(list(INDUSTRIES.keys()))
        keyword = random.choice(INDUSTRIES[industry])
        description = generate_business_description(industry, keyword)

        domain_list = []
        attempts = 0

        while len(domain_list) < DOMAINS_PER_SAMPLE and attempts < 20:
            strategy = random.choice(STRATEGIES)
            try:
                domain = (
                    strategy(keyword)
                    if "acronym" not in strategy.__name__
                    else strategy(industry, keyword)
                )
                if is_safe(domain):
                    confidence = score_domain(domain, keyword)
                    domain_list.append({"domain": domain, "confidence": confidence})
            except Exception:
                continue
            attempts += 1

        if domain_list:
            dataset.append({"input": description, "output": domain_list})

    return dataset


if __name__ == "__main__":

    dataset = generate_dataset()
    save_dataset(dataset, filename=OUTPUT_FILE)
