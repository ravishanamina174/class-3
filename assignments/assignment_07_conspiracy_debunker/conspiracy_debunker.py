"""
Assignment 7: Conspiracy Theory Debunker
Zero-Shot + Chain of Thought - Analyze and debunk misinformation

Your mission: Combat misinformation by analyzing conspiracy theories
with clear instructions and step-by-step logical reasoning!
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


@dataclass
class DebunkAnalysis:
    conspiracy_text: str
    main_claims: List[str]
    logical_flaws: List[str]
    reasoning_chain: List[str]
    psychological_appeal: str
    debunking_summary: str
    reliable_sources: List[str]
    confidence_score: float


class ConspiracyDebunker:
    """
    AI-powered conspiracy theory analyzer using zero-shot + CoT.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.analysis_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot CoT chain for conspiracy analysis.

        Combine clear instructions with "let's think step by step"
        """

        template = PromptTemplate.from_template(
            """Analyze this conspiracy theory respectfully but critically.

[TODO: Add zero-shot instructions for:
- Claim extraction
- Logical flaw identification  
- Step-by-step reasoning (CoT)
- Respectful tone]

Theory: {conspiracy_text}

Let's think step by step:"""
        )

        # TODO: Set up the chain
        pass

    def debunk(self, conspiracy_text: str) -> DebunkAnalysis:
        """
        TODO #2: Analyze and debunk conspiracy theory.

        Use zero-shot for novel analysis + CoT for reasoning
        """

        # TODO: Implement analysis combining both techniques

        return DebunkAnalysis(
            conspiracy_text=conspiracy_text,
            main_claims=[],
            logical_flaws=[],
            reasoning_chain=[],
            psychological_appeal="",
            debunking_summary="",
            reliable_sources=[],
            confidence_score=0.0,
        )


def test_debunker():
    debunker = ConspiracyDebunker()

    test_theories = [
        "Birds aren't real - they're government surveillance drones. Notice how they sit on power lines to recharge?",
        "The moon landing was filmed in a Hollywood studio. The flag waves despite no atmosphere!",
        "Chemtrails from planes are mind control chemicals. Normal contrails disappear quickly but these linger!",
    ]

    print("🤔 CONSPIRACY THEORY DEBUNKER 🤔")
    print("=" * 70)

    for theory in test_theories:
        result = debunker.debunk(theory)
        print(f'\nTheory: "{theory[:60]}..."')
        print(f"Main Claims: {len(result.main_claims)} identified")
        print(f"Logical Flaws: {len(result.logical_flaws)} found")
        print(f"Confidence: {result.confidence_score:.0%}")
        print("-" * 70)


if __name__ == "__main__":
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_debunker()
