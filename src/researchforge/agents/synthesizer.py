
import os
import time

from google import genai


class Synthesizer:
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

    def synthesize(self, research_question: str, research_results: list[dict]) -> str:

        evidence_context = []

        for i, result in enumerate(research_results):
            evidence_context.append(
                f"\nResearch subquestion {i + 1}: {result['subquestion']}\n"
            )

            for item in result["evidence"]:
                evidence_context.append(
                    f"""
Paper: {item.chunk.title}
Paper ID: {item.chunk.paper_id}
Similarity score: {item.score:.3f}
Evidence:
{item.chunk.text}
"""
                )

        context = "\n".join(evidence_context)

        prompt = f"""
You are the Synthesizer Agent of ResearchForge, an academic research assistant.

Your task is to synthesize findings from the supplied research evidence.

Original research question:
{research_question}

Retrieved research evidence:
{context}

Instructions:
1. Identify the major findings supported by the evidence.
2. Group related findings into meaningful themes.
3. Identify agreements and disagreements between papers.
4. Discuss limitations and unresolved questions where supported.
5. Write a coherent, technically detailed synthesis.
6. Cite claims using the supplied paper titles and paper IDs.
7. Never invent papers, citations, experimental results or findings.
8. If the evidence does not support a claim, explicitly state that.
9. Do not treat similarity scores as measures of factual reliability.

Structure your response using these sections:

1. Overview
2. Key Findings
3. Agreements and Disagreements
4. Limitations and Research Gaps
5. Conclusion

Only use the supplied evidence. Do not answer from your general knowledge.
"""

        for model in self.models:
            try:
                print(f"Trying model: {model}")

                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if not response.text:
                    raise ValueError("Empty synthesis response.")

                self.model = model
                print(f"Successful model: {model}")

                return response.text

            except Exception as error:
                print(f"{model} failed: {error}")
                time.sleep(1)

        raise RuntimeError("All available Gemini models failed to generate synthesis.")
