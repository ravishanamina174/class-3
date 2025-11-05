"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought - Decode alien messages using examples and reasoning

Your mission: First contact! Decode alien communications using pattern
recognition and logical deduction!
"""

import os
import json
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class Translation:
    alien_text: str
    human_text: str
    confidence: float
    reasoning_steps: List[str]
    cultural_notes: str


class AlienTranslator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.translation_examples = self._load_examples()
        self.decoder_chain = None
        self._setup_chains()

    def _load_examples(self) -> List[dict]:
        examples = [
            {
                "alien": "◈◈◈ ▲▲ ●",
                "reasoning": "Step 1: ◈ repeated 3 times indicates quantity (3).\nStep 2: ▲▲ repeated indicates object type (ship).\nStep 3: ● marks an action/approach.\nConclusion: 'Three ships approaching'",
                "translation": "Three ships approaching",
                "pattern": "quantity-object-action",
            },
            {
                "alien": "♦♦ ◯◯◯ ▼",
                "reasoning": "Step 1: ♦♦ indicates location.\nStep 2: ◯◯◯ indicates quantity (3 crew).\nStep 3: ▼ means departing.\nConclusion: 'Three crew departing the harbor'",
                "translation": "Three crew departing the harbor",
                "pattern": "location-quantity-action",
            },
            {
                "alien": "★★★ ▲ ◆◆",
                "reasoning": "Step 1: ★★★ indicates urgency.\nStep 2: ▲ is vessel.\nStep 3: ◆◆ means damaged.\nConclusion: 'High alert: vessel damaged'",
                "translation": "High alert: vessel damaged",
                "pattern": "urgency-object-state",
            },
            {
                "alien": "△△ △ ◈●",
                "reasoning": "Step 1: △△ = two small units.\nStep 2: △ = one unit.\nStep 3: ◈● = observe.\nConclusion: 'Two drones observing the area'",
                "translation": "Two drones observing the area",
                "pattern": "quantity-object-action",
            },
        ]
        return examples

    def _setup_chains(self):
        example_prompt = PromptTemplate.from_template(
            """Alien: {alien}
Reasoning: {reasoning}
Translation: {translation}
Pattern: {pattern}"""
        )

        prefix = (
            "You are an expert xenolinguist. Use examples to decode alien symbol strings.\n"
            "Follow the reasoning style and produce an output JSON with keys: human_text, confidence, reasoning_steps (list), cultural_notes.\n"
        )

        suffix = (
            "Now decode the following alien message:\n"
            "Alien: {alien}\n"
            "Output JSON as described above."
        )

        self.decoder_chain = (
            FewShotPromptTemplate(
                examples=self.translation_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["alien"],
            )
            | self.llm
            | StrOutputParser()
        )

    def translate(self, alien_message: str) -> Translation:
        try:
            raw = self.decoder_chain.invoke({"alien": alien_message})
            try:
                parsed = json.loads(raw)
            except:
                block = raw[raw.find("{"): raw.rfind("}") + 1]
                parsed = json.loads(block) if block.strip() else None

            if parsed:
                return Translation(
                    alien_text=alien_message,
                    human_text=parsed.get("human_text") or parsed.get("translation", ""),
                    confidence=float(parsed.get("confidence", 0.6)),
                    reasoning_steps=parsed.get("reasoning_steps", []),
                    cultural_notes=parsed.get("cultural_notes", ""),
                )
        except:
            pass

        mapping = {}
        for ex in self.translation_examples:
            for a, t in zip(ex["alien"].split(), ex["translation"].split()):
                mapping.setdefault(a, t)

        translated = [mapping.get(tok, "?") for tok in alien_message.split()]
        return Translation(
            alien_text=alien_message,
            human_text=" ".join(translated),
            confidence=0.3,
            reasoning_steps=["Fallback heuristic mapping used."],
            cultural_notes="Symbols guessed from examples",
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
        print("Reasoning:")
        for step in result.reasoning_steps:
            print(f"  - {step}")
        print("-" * 70)


if __name__ == "__main__":
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_translator()
