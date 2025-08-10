import itertools
import pandas as pd
from dataclasses import dataclass

from typing import List, Optional


@dataclass
class PromptSpec:
    business: str
    style: str
    tld: Optional[str]
    length: str
    language: str
    noise: Optional[str]

    def render_prompt(self, template: str) -> str:
        return template.format(
            business_description=self.business,
            style=self.style,
            tld=self.tld or "",
            length=self.length,
            language=self.language,
            noise=self.noise or "",
        )


def build_prompt_matrix(
    industries: List[str],
    styles: List[str],
    lengths: List[str],
    tlds: List[Optional[str]],
    languages: List[str],
    weirdness_levels: List[str],
    template: str,
) -> pd.DataFrame:
    """Creates a DataFrame with all combinations of prompt parameters."""
    rows = []
    prod = itertools.product(
        industries, styles, lengths, tlds, languages, weirdness_levels
    )
    for ind, style, length, tld, lang, weird in prod:
        if length == "short":
            biz = f"{ind}"
        elif length == "long":
            biz = f"{ind} that focuses on artisanal small-batch production and sustainability."
        else:  # verbose
            biz = f"{ind} operating globally, with subscription services and custom collaborations."

        if weird == "absurd":
            biz += " It also designs hats for cats and offers intergalactic shipping."
        elif weird == "creative":
            biz += " It has an experimental brand voice and bold aesthetics."

        spec = PromptSpec(
            business=biz, style=style, tld=tld, length=length, language=lang, noise=None
        )
        rows.append(
            {
                "business": biz,
                "style": style,
                "tld": tld,
                "length": length,
                "language": lang,
                "weirdness": weird,
                "prompt": spec.render_prompt(template),
            }
        )
    return pd.DataFrame(rows)
