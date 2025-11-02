"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought - Decode alien messages using examples and reasoning

Your mission: First contact! Decode alien communications using pattern
recognition and logical deduction!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


@dataclass
class Translation:
    alien_text: str
    human_text: str
    confidence: float
    reasoning_steps: List[str]
    cultural_notes: str


class AlienTranslator:
    """
    AI-powered alien language translator using few-shot examples and CoT reasoning.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.translation_examples = self._load_examples()
        self.decoder_chain = None
        self._setup_chains()

    def _load_examples(self) -> List[dict]:
        """
        TODO #1: Create example alien translations with reasoning.

        Include: symbols, translation, step-by-step decoding logic
        """

        examples = [
            {
                "alien": "◈◈◈ ▲▲ ●",
                "reasoning": "Step 1: ◈◈◈ appears to be quantity (3 symbols)\nStep 2: ▲▲ represents object type\nStep 3: ● is singular marker\nStep 4: Pattern suggests 'three ships approaching'",
                "translation": "Three ships approaching",
                "pattern": "quantity-object-verb",
            },
            # TODO: Add more examples with reasoning chains
        ]

        return examples

    def _setup_chains(self):
        """
        TODO #2: Create few-shot CoT chain for translation.

        Combine pattern examples with reasoning steps.
        """

        # TODO: Create combined few-shot + CoT template
        pass

    def translate(self, alien_message: str) -> Translation:
        """
        TODO #3: Translate alien message using examples and reasoning.

        Args:
            alien_message: Message to decode

        Returns:
            Translation with reasoning
        """

        # TODO: Apply few-shot patterns and CoT reasoning

        return Translation(
            alien_text=alien_message,
            human_text="",
            confidence=0.0,
            reasoning_steps=[],
            cultural_notes="",
        )


def test_translator():
    translator = AlienTranslator()

    test_messages = ["◈◈◈◈◈ ▲▲▲ ● ◆", "♦♦ ◯◯◯ ▼ ★★★★", "△△△ ◈ ■■ ◆◆◆"]

    print("👽 ALIEN LANGUAGE TRANSLATOR 👽")
    print("=" * 70)

    for msg in test_messages:
        result = translator.translate(msg)
        print(f"\nAlien: {msg}")
        print(f"Translation: {result.human_text}")
        print(f"Confidence: {result.confidence:.0%}")
        print("-" * 70)


if __name__ == "__main__":
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_translator()
