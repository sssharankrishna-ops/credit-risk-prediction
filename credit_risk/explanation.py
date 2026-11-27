"""Optional local Hugging Face explanation generation with a safe fallback."""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _local_generator():
    model_path = os.getenv("CREDIT_RISK_EXPLANATION_MODEL")
    if not model_path:
        return None

    from langchain_core.prompts import PromptTemplate
    from transformers import pipeline

    prompt = PromptTemplate.from_template(
        "In one plain-language sentence, explain this credit risk estimate without "
        "making a lending decision. Use only these factors: {factors}. "
        "Risk category: {category}."
    )
    generator = pipeline(
        "text2text-generation",
        model=model_path,
        tokenizer=model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    return prompt, generator


def explain_prediction(prediction: dict) -> str:
    factors = prediction["key_factors"]
    default_text = "Risk factors: " + "; ".join(
        f"{factor['label']} {factor['direction']} estimated risk"
        for factor in factors
    ) + "."
    try:
        integration = _local_generator()
        if integration is None:
            return default_text
        prompt, generator = integration
        response = generator(
            prompt.format(
                factors=", ".join(
                    f"{factor['label']} ({factor['direction']} risk)"
                    for factor in factors
                ),
                category=prediction["risk_category"],
            ),
            max_new_tokens=80,
            do_sample=False,
        )
        return response[0]["generated_text"].strip() or default_text
    except Exception:
        return default_text
