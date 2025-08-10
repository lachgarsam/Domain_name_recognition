import os
import logging

from orchestrator import EdgeCaseTester
from generation import GenerationConfig
from prompting import build_prompt_matrix
from analysis import (
    compute_failure_metrics,
    plot_failure_rates,
)

logger = logging.getLogger(__name__)


TEMPLATE = """<|system|>
You are a domain name generator. Suggest domain names based on the business description. Output a single JSON array of objects, each with "domain" and "confidence" keys. Do not add any extra text.

<|user|>
Business: {business_description}
Style: {style}

<|assistant|>
"""

INDUSTRIES = ["A cozy organic coffee shop", "A startup that sells hats for cats"]
STYLES = ["short one-word names", "brandable modern names"]
LENGTHS = ["short", "long"]
TLDS = [None, ".com", ".io"]
LANGUAGES = ["en"]
WEIRDNESS = ["normal", "creative"]


def main(model_path: str, output_dir: str):

    logger.info("Building prompt matrix...")
    df_prompts = build_prompt_matrix(
        INDUSTRIES, STYLES, LENGTHS, TLDS, LANGUAGES, WEIRDNESS, TEMPLATE
    )
    logger.info(f"   Generated {len(df_prompts)} unique prompts.")

    tester = EdgeCaseTester(
        model_path=model_path,
        output_dir=output_dir,
    )
    gen_cfg = GenerationConfig(batch_size=1)

    logger.info("\nRunning tests...")
    results = tester.run_matrix(df_prompts, cfg=gen_cfg)

    logger.info("\nExporting reports...")
    rates = compute_failure_metrics(results)
    logger.info(f"   Failure Rates: {rates}")

    plot_failure_rates(rates, os.path.join(output_dir, "failure_rates.png"))

    logger.info(f"\n All tasks complete. Check the '{output_dir}' directory.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True, help="Path to your fine-tuned model folder"
    )
    parser.add_argument(
        "--output_dir",
        default="edge_runs",
        help="folder to save edgecas results",
    )
    args = parser.parse_args()

    main(args.model_path, args.output_dir)
