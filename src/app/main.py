import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from peft import PeftModel

from data_models import DomainSuggestionResponse, SuggestRequest
from utils import (
    get_device,
    generate_domain_suggestions,
    clean_raw_response,
    KoalaModerationFilter,
)

lifespan_objects = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan event handler that runs at application startup and shutdown.

    On startup:
        - Loads environment variables
        - Loads base language model and fine-tuned PEFT model onto the appropriate device
        - Initializes tokenizer
        - Initializes content moderation filter

    On shutdown:
        - Clears all loaded resources to free memory

    Args:
        app (FastAPI): The FastAPI application instance

    Yields:
        AsyncGenerator[None, None]: None during the lifespan of the app
    """

    # from dotenv import load_dotenv

    # load_dotenv()

    MODEL_PATH = os.getenv("MODEL_PATH")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    lifespan_objects["device"] = get_device()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        return_dict=True,
        torch_dtype=(
            torch.bfloat16 if lifespan_objects["device"] == "mps" else torch.float16
        ),
        use_auth_token=HUGGINGFACE_TOKEN
    )

    lifespan_objects["finetuned_model"] = PeftModel.from_pretrained(
        base_model, MODEL_PATH
    )

    lifespan_objects["finetuned_model"].to(lifespan_objects["device"])

    lifespan_objects["finetuned_model"].eval()

    lifespan_objects["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)

    lifespan_objects["moderator"] = KoalaModerationFilter(
        device=lifespan_objects["device"]
    )

    yield

    lifespan_objects.clear()


app = FastAPI(lifespan=lifespan)


@app.post(
    "/suggest_domains",
    response_model=DomainSuggestionResponse,
    response_model_exclude_none=True,
)
async def suggest_domains(payload: SuggestRequest) -> DomainSuggestionResponse:
    """
    API endpoint to suggest domain names based on a business description.

    Steps:
    - Checks description for inappropriate content using a moderation filter
    - If safe, generates domain name suggestions with confidences from the language model
    - Parses and returns suggestions or returns blocked status if flagged

    Args:
        payload (SuggestRequest): Request payload containing business description

    Returns:
        DomainSuggestionResponse: Contains domain suggestions or block message
    """
    description = payload.business_description
    model, tokenizer, device = (
        lifespan_objects["finetuned_model"],
        lifespan_objects["tokenizer"],
        lifespan_objects["device"],
    )
    moderator = lifespan_objects["moderator"]

    if moderator.check(description):
        return DomainSuggestionResponse(
            suggestions=[],
            status="blocked",
            message="Request contains inappropriate content",
        )

    raw_response = await generate_domain_suggestions(
        model, tokenizer, device, description
    )

    cleaned = clean_raw_response(raw_response)

    try:
        suggestions_list = json.loads(cleaned)  # Now a list of dicts
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse suggestions: {e}\nRaw: {cleaned}")
    return DomainSuggestionResponse(suggestions=suggestions_list, status="success")


# Uncomment to run locally using uvicorn
# [Errno 48]: address already in use: lsof -i tcp:8080 ; kill -9 <PID>
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
