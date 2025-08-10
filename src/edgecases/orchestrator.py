import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import time
import json
import logging
from dataclasses import dataclass, asdict
import pandas as pd

from typing import List, Dict, Any, Optional

from generation import DomainGenerator, GenerationConfig
from analysis import (
    detect_offensive,
    detect_repetition,
    detect_tld_mismatch,
    detect_nonsensical,
)
from utils import extract_json_array, get_device
from evaluation.llm_judge import LLMDomainJudge


logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    prompt_meta: Dict[str, Any]
    raw_output: str
    parsed: Optional[Any]
    auto_issues: Dict[str, Any]
    judge_result: Optional[Dict[str, Any]] = None
    timestamp: float = time.time()


class EdgeCaseTester:
    def __init__(
        self,
        model_path: str,
        output_dir: str = "edge_runs",
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.gen = DomainGenerator(model_path)
        self.judge = LLMDomainJudge(
            model_id="mistralai/Mistral-7B-Instruct-v0.2", device=get_device()
        )

    def run_matrix(
        self, df_prompts: pd.DataFrame, cfg: GenerationConfig
    ) -> List[RunResult]:
        results: List[RunResult] = []
        prompts = df_prompts["prompt"].tolist()

        logger.info(
            f"Starting generation on {len(prompts)} prompts with batch size {cfg.batch_size}..."
        )
        raw_outputs = self.gen.generate_batch(prompts, cfg)

        logger.info("Analyzing results...")
        for i, raw_out in enumerate(raw_outputs):
            row_meta = df_prompts.iloc[i].to_dict()
            parsed = extract_json_array(raw_out)

            auto_issues = {}
            if detect_offensive(raw_out)[0]:
                auto_issues["offensive"] = True
            if detect_repetition(raw_out)[0]:
                auto_issues["repetition"] = True
            if detect_nonsensical(parsed)[0]:
                auto_issues["nonsensical"] = True
            if detect_tld_mismatch(parsed, row_meta.get("tld"))[0]:
                auto_issues["tld_mismatch"] = True
            judge_res = self.judge.rate_predictions(row_meta["business"], parsed)

            results.append(
                RunResult(
                    prompt_meta=row_meta,
                    raw_output=raw_out,
                    parsed=parsed,
                    auto_issues=auto_issues,
                    judge_result=judge_res,
                )
            )

        self.save_results(results, "final_results.json")
        return results

    def save_results(self, results: List[RunResult], filename: str):
        path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving {len(results)} results to {path}")
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
