# Domain Name Suggestion



**DomainNameSuggester** is a modular training & evaluation toolbox for teaching LLMs to suggest domain names from short business descriptions.  
It includes:

- dataset generation (rule-based + LLM-based synthetic generator),
- multiple fine-tuning strategies (Full, LoRA, QLoRA, Prefix, IA³) implemented with PEFT / TRL,
- Optuna HPO runner with MLflow tracking,
- an evaluator that uses a separate LLM as a judge,
- a FastAPI service for inference with safety filtering and edge-case handling.


## Project structure
```
📂 A high-level overview of the project's directory 
    Domain_name_suggestion/
├── README.md
├── Dockerfile
├── requirements.txt
├── models/                      # Saved models, adapters, and checkpoints
│   └── best_model_from_hpo/
└── src/
    ├── app/
    │   ├── .env
    │   ├── main.py              # FastAPI app entrypoint
    │   ├── data_models.py       # Pydantic request/response schemas
    │   └── utils.py             # API helpers: model loader, safety checks, etc.
    │
    ├── data/
    │   ├── config.py
    │   ├── llm_generated_dataset.py       # LLM-driven synthetic data generation
    │   ├── rule_based_dataset_generation.py # Rule Based Generated dataset
    │   ├── rule_based_generation_strategies.py
    │   └── utils.py
    │
    ├── edgecases/
    │   ├── analysis.py          # Post-run failure analysis and plots
    │   ├── generation.py        # Prompt matrix generator
    │   ├── orchestrator.py      # Run batches of edge-case tests
    │   ├── prompting.py         # Templates & perturbations for tests
    │   ├── run_tests.py         # CLI to run the edge-case suite
    │   └── utils.py
    │
    ├── evaluation/
    │   ├── evaluate.py          # Batch evaluation (suggester + judge)
    │   ├── llm_judge.py         # LLM judge (rate/rank suggestions)
    │   └── test_llm_judge.py
    │
    └── experiments/
        ├── data_processing.py   # Dataset mapping / prompt formatting / tokenization
        ├── finetuner.py         # FineTuner class (strategy pattern)
        ├── tuning_strategies.py # LoRA / QLoRA / Prefix / IA3 / Full strategy classes
        ├── run_hpo.py           # Optuna + MLflow orchestration + trial checkpoints
        └── utils.py             # helpers: yaml loader, logging
```

## Build and Run Docker container

```
docker build -t domain-suggester:latest -f src/Dockerfile .
```

```
docker run --rm -it \
  -p 8080:8080 \
  -v "$(pwd)/models:/models" \
  -e HUGGINGFACE_TOKEN="token" \
  -e MODEL_PATH="/models/my-finetuned" \
  domain-suggester:latest
```
Please make sure to Mount the model directory and pass HF token and model path.