
import os
import json
import time

from google import genai
from google.genai import types


class Critic:
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

    def evaluate(
        self,
        research_question: str,
        synthesis: str,
        research_results: list[dict],
        fact_check_results: list[dict]
    ) -> dict:

        evidence_context = []

        for result in research_results:
            evidence_context.append(
                f"\nSubquestion: {result['subquestion']}\n"
            )

            for item in result["evidence"]:
                evidence_context.append(
                    f"""
Paper: {item.chunk.title}
Paper ID: {item.chunk.paper_id}
Evidence:
{item.chunk.text}
"""
                )

        verdict_counts = {
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0
        }

        for claim in fact_check_results:
            verdict = claim.get("verdict")

            if verdict in verdict_counts:
                verdict_counts[verdict] += 1

        prompt = f"""
You are the Critic Agent of ResearchForge.

Your responsibility is to evaluate the quality, completeness and
evidence-grounding of a research synthesis.

ORIGINAL RESEARCH QUESTION:
{research_question}

RESEARCH SYNTHESIS:
{synthesis}

RETRIEVED RESEARCH EVIDENCE:
{"".join(evidence_context)}

FACT-CHECK RESULTS:
{json.dumps(fact_check_results, indent=2)}

FACT-CHECK SUMMARY:
{json.dumps(verdict_counts, indent=2)}

Evaluate the synthesis using ONLY the supplied information.

Consider:

1. Relevance:
   Does the synthesis address the original research question?

2. Evidence coverage:
   Are the important aspects of the research question adequately covered?

3. Evidence grounding:
   Are the conclusions supported by the retrieved evidence?
   Pay particular attention to unsupported and partially supported claims.

4. Depth:
   Is the technical discussion sufficiently detailed?

5. Research gaps:
   Are important unanswered aspects or missing perspectives identified?

6. Contradictions:
   Are disagreements between papers adequately discussed?

7. Factual reliability:
   Does the fact-check report reveal claims that lack sufficient evidence?

Return a JSON object with exactly this structure:

{{
    "scores": {{
        "relevance": 1,
        "evidence_coverage": 1,
        "evidence_grounding": 1,
        "depth": 1,
        "overall": 1
    }},
    "strengths": [],
    "weaknesses": [],
    "missing_aspects": [],
    "follow_up_questions": [],
    "recommendation": "accept",
    "reason": "Explanation of the recommendation"
}}

Scoring:
- Use integer scores from 1 to 10.
- 1 means extremely poor.
- 10 means excellent.

Recommendation:
- "accept": The synthesis adequately addresses the question
  and its important claims are sufficiently supported.
- "research_more": Important evidence gaps or unsupported claims
  remain that additional research could reasonably address.

Rules:
- Do not invent findings or citations.
- Do not use external knowledge.
- Do not reward a synthesis merely for being lengthy.
- Treat unsupported claims as serious weaknesses.
- Partially supported claims must be examined carefully.
- Follow-up questions must address actual gaps.
- Return valid JSON only.
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
                    raise ValueError("Empty critic response.")

                result = json.loads(response.text)

                scores = result.get("scores")

                if not isinstance(scores, dict):
                    raise ValueError("Missing scores.")

                score_names = [
                    "relevance",
                    "evidence_coverage",
                    "evidence_grounding",
                    "depth",
                    "overall"
                ]

                for name in score_names:
                    value = scores.get(name)

                    if not isinstance(value, int) or not 1 <= value <= 10:
                        raise ValueError(f"Invalid score: {name}")

                for field in [
                    "strengths",
                    "weaknesses",
                    "missing_aspects",
                    "follow_up_questions"
                ]:
                    if not isinstance(result.get(field), list):
                        raise ValueError(f"Invalid field: {field}")

                if result.get("recommendation") not in {
                    "accept",
                    "research_more"
                }:
                    raise ValueError("Invalid recommendation.")

                if not isinstance(result.get("reason"), str):
                    raise ValueError("Missing recommendation reason.")

                # Deterministic acceptance safeguards
                if verdict_counts["unsupported"] > 0:
                    result["recommendation"] = "research_more"
                    result["reason"] = (
                        f"{verdict_counts['unsupported']} claims were "
                        "classified as unsupported by the FactChecker. "
                        "The report requires further verification."
                    )

                elif (
                    verdict_counts["partially_supported"] > 0
                    and result["scores"]["evidence_grounding"] < 8
                ):
                    result["recommendation"] = "research_more"
                    result["reason"] = (
                        "Some claims are only partially supported, "
                        "and evidence grounding is insufficient."
                    )

                elif (
                    result["missing_aspects"]
                    and result["follow_up_questions"]
                    and result["scores"]["evidence_coverage"] < 9
                ):
                    result["recommendation"] = "research_more"
                    result["reason"] = (
                        "Important aspects remain missing, and actionable "
                        "follow-up questions exist."
                    )

                self.model = model
                print(f"Successful model: {model}")

                return result

            except Exception as error:
                print(f"{model} failed: {error}")
                time.sleep(1)

        raise RuntimeError(
            "All available Gemini models failed to evaluate the synthesis."
        )
