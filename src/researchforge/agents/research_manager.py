import os
import json

from google import genai
from google.genai import types


class ResearchManager:
    def __init__(self, model="gemini-3.5-flash-lite"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text

    def decompose_query(self, research_question: str) -> list[str]:
        prompt = f"""
You are the Research Manager of ResearchForge, an AI research assistant.

Your task is to decompose a research question into focused subquestions
that can be independently investigated by research agents.

Instructions:
1. Understand the central research question.
2. Identify its major dimensions and aspects.
3. Generate between 5 and 8 subquestions depending on complexity.
4. Ensure the subquestions collectively cover the original question.
5. Avoid redundant or overlapping subquestions.
6. Include relevant technical, methodological, and evaluation aspects.
7. Each subquestion must be independently researchable.
8. Do not answer the questions.

Research question:
{research_question}

Return a JSON object with this structure:
{{
    "subquestions": [
        "First subquestion",
        "Second subquestion"
    ]
}}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        result = json.loads(response.text)
        subquestions = result.get("subquestions")

        if not isinstance(subquestions, list) or not 5 <= len(subquestions) <= 8:
            raise ValueError("Gemini returned an invalid number of subquestions.")

        if not all(isinstance(q, str) and q.strip() for q in subquestions):
            raise ValueError("Gemini returned invalid subquestions.")

        return subquestions