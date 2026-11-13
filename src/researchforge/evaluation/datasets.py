
from typing import TypedDict


class BenchmarkQuestion(TypedDict):
    question: str
    category: str
    expected_aspects: list[str]


BENCHMARK_DATASET: list[BenchmarkQuestion] = [
    {
        "question": "How do Retrieval-Augmented Generation systems improve the factual reliability of Large Language Models?",
        "category": "RAG",
        "expected_aspects": [
            "Retrieval and generation architecture",
            "Dense and sparse retrieval methods",
            "Knowledge grounding and hallucination reduction",
            "Retrieval quality and relevance",
            "Limitations of RAG systems",
            "Evaluation methods for RAG"
        ]
    },
    {
        "question": "How has Reinforcement Learning from Human Feedback evolved, and what are its limitations?",
        "category": "RLHF",
        "expected_aspects": [
            "Human preference data collection",
            "Reward model training",
            "Proximal Policy Optimization",
            "Direct Preference Optimization",
            "Alignment and preference learning",
            "Limitations and training challenges"
        ]
    },
    {
        "question": "What are the major approaches to few-shot learning, and how does Model-Agnostic Meta-Learning work?",
        "category": "Few-shot Learning",
        "expected_aspects": [
            "Definition of few-shot learning",
            "Metric-based approaches",
            "Model-based approaches",
            "Optimization-based approaches",
            "MAML inner and outer optimization loops",
            "Advantages and limitations of MAML"
        ]
    },
    {
        "question": "How do Transformer architectures achieve effective sequence modeling compared with recurrent neural networks?",
        "category": "Transformers",
        "expected_aspects": [
            "Self-attention mechanism",
            "Multi-head attention",
            "Positional encoding",
            "Transformer encoder and decoder",
            "Comparison with recurrent neural networks",
            "Computational limitations and scalability"
        ]
    },
    {
        "question": "What are the major challenges in training and deploying Large Language Models?",
        "category": "LLMs",
        "expected_aspects": [
            "Pretraining objectives and datasets",
            "Computational requirements",
            "Fine-tuning and alignment",
            "Hallucinations and factual reliability",
            "Inference efficiency",
            "Safety and deployment challenges"
        ]
    }
]


def get_benchmark_dataset() -> list[BenchmarkQuestion]:
    """Return all benchmark research questions."""
    return BENCHMARK_DATASET.copy()


def get_questions_by_category(category: str) -> list[BenchmarkQuestion]:
    """Return benchmark questions belonging to a category."""
    return [
        question
        for question in BENCHMARK_DATASET
        if question["category"].lower() == category.lower()
    ]
