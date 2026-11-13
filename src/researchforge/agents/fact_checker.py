
import os
import json
import time

from google import genai
from google.genai import types


class FactChecker:
    MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
        "gemini-4-hydrogen"
    ]

    def __init__(self, model=None):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=api_key)

        self.models = (
            [model] + [m for m in self.MODELS if m != model]
            if model else self.MODELS.copy()
        )

        self.model = None

    def check(self, synthesis: str, research_results: list[dict]) -> list[dict]:

        evidence_context = []

        for result in research_results:
            for item in result["evidence"]:
                evidence_context.append({
                    "paper_id": item.chunk.paper_id,
                    "title": item.chunk.title,
                    "text": item.chunk.text
                })

        prompt = f"""
You are the Fact Checker Agent of ResearchForge.

Your task is to verify factual claims made in a research synthesis
using ONLY the evidence provided below.

RESEARCH SYNTHESIS:
{synthesis}

AVAILABLE EVIDENCE:
{json.dumps(evidence_context, indent=2)}

Instructions:

1. Extract the important factual claims from the synthesis.
2. Check each claim against the available evidence.
3. Assign exactly one verdict:
   - supported
   - partially_supported
   - unsupported
4. A claim is supported only if the evidence directly establishes it.
5. A claim is partially_supported if the evidence establishes only
   some parts of it.
6. A claim is unsupported if the evidence does not establish it.
7. Do not use external knowledge to verify claims.
8. Do not assume that a paper supports a claim merely because
   its title or abstract sounds relevant.
9. For every claim, identify the specific evidence that supports
   or contradicts it, if any.
10. Never invent paper IDs, quotations, or evidence.

Return valid JSON in exactly this format:

{{
    "claims": [
        {{
            "claim": "Factual claim from the synthesis",
            "verdict": "supported",
            "explanation": "Why this verdict was assigned",
            "evidence": [
                {{
                    "paper_id": "Paper ID",
                    "quote": "Exact supporting text from the evidence"
                }}
            ]
        }}
    ]
}}

If no evidence supports a claim, return an empty evidence list.
"""

        for model in self.models:
            try:
                print(f"Trying model: {model}")

                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                if not response.text:
                    raise ValueError("Empty fact-checking response.")

                result = json.loads(response.text)

                if not isinstance(result, dict):
                    raise ValueError("Invalid response format.")

                claims = result.get("claims")

                if not isinstance(claims, list):
                    raise ValueError("Missing claims list.")

                valid_verdicts = {
                    "supported",
                    "partially_supported",
                    "unsupported"
                }

                for claim in claims:
                    if not isinstance(claim, dict):
                        raise ValueError("Invalid claim format.")

                    if claim.get("verdict") not in valid_verdicts:
                        raise ValueError("Invalid verdict.")

                    if not isinstance(claim.get("evidence"), list):
                        raise ValueError("Invalid evidence format.")

                self.model = model
                print(f"Successful model: {model}")

                return claims

            except Exception as error:
                print(f"{model} failed: {error}")
                time.sleep(1)

        raise RuntimeError("All available Gemini models failed fact-checking.")
