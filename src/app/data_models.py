from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SuggestRequest(BaseModel):
    business_description: str = Field(
        ..., example="An organic coffee shop in downtown area"
    )


class DomainSuggestion(BaseModel):
    domain: str
    confidence: Optional[float] = None


class DomainSuggestionResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: Optional[str] = None
