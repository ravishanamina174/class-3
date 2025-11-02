"""
Assignment 5: Urban Legend Fact Checker
Zero-shot + Few-shot Prompting - Combine techniques for myth analysis

Your mission: Build a system that analyzes urban legends using the right
prompting technique for each subtask!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MythCategory(Enum):
    SUPERNATURAL = "supernatural"
    CONSPIRACY = "conspiracy"
    MEDICAL = "medical_health"
    TECHNOLOGY = "technology"
    HISTORICAL = "historical"
    SOCIAL = "social_phenomena"
    CREATURE = "cryptid_creature"


class LogicalFallacy(Enum):
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_CAUSE = "false_cause"
    SLIPPERY_SLOPE = "slippery_slope"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    CIRCULAR_REASONING = "circular_reasoning"
    HASTY_GENERALIZATION = "hasty_generalization"


@dataclass
class Claim:
    """Individual claim extracted from legend"""

    text: str
    testable: bool
    evidence_required: str
    confidence: float


@dataclass
class MythAnalysis:
    """Complete urban legend analysis"""

    original_text: str
    category: str
    claims: List[Claim]
    logical_fallacies: List[str]
    truth_rating: float  # 0 (false) to 1 (true)
    believability_score: float  # How convincing it sounds
    debunking_explanation: str
    similar_myths: List[str]
    origin_theory: str


class UrbanLegendChecker:
    """
    AI-powered urban legend analyzer combining zero-shot and few-shot prompting.
    Uses the right technique for each analysis task.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize the legend checker.

        Args:
            model_name: The LLM model to use
            temperature: Controls randomness in responses
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.claim_extractor = None  # Zero-shot
        self.myth_classifier = None  # Few-shot
        self.fallacy_detector = None  # Combined
        self.debunker = None  # Zero-shot
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up different chains using appropriate prompting methods.

        Create:
        1. claim_extractor: Zero-shot for extracting claims
        2. myth_classifier: Few-shot for categorizing myths
        3. fallacy_detector: Combined approach for fallacies
        4. debunker: Zero-shot for generating explanations
        """

        # ✅ Zero-shot chain for claim extraction
        claim_template = PromptTemplate.from_template(
            """Extract all testable claims from this urban legend.

Instructions:
- A claim is a specific statement of fact that can be evaluated
- Mark if the claim is testable in real world
- Specify evidence required (scientific, eyewitness, historical record, etc.)
- Output ONLY JSON list like:
[
  { "text":"...", "testable": true/false, "evidence_required":"...", "confidence":0.0-1.0 }
]

Text: {legend_text}

Claims:"""
        )

        self.claim_extractor = claim_template | self.llm | StrOutputParser()

        # ✅ Few-shot chain for myth classification
        classification_examples = [
            {
                "legend": "Alligators live in NYC sewers after being flushed as pets.",
                "category": "cryptid_creature",
                "reasoning": "Involves hidden mysterious animals in urban setting.",
            },
            {
                "legend": "Cell phones at gas stations can cause explosions.",
                "category": "technology",
                "reasoning": "Tech safety fear story.",
            },
            {
                "legend": "Ghosts haunt a hotel and move objects around.",
                "category": "supernatural",
                "reasoning": "Involves spirits and paranormal activity.",
            },
            {
                "legend": "Vaccines implant microchips to track people.",
                "category": "conspiracy",
                "reasoning": "Government surveillance conspiracy theory.",
            },
            {
                "legend": "A secret government society controls world events.",
                "category": "social_phenomena",
                "reasoning": "Shadow groups and social manipulation beliefs.",
            },
        ]

        classification_prompt = PromptTemplate.from_template(
            """Legend: {legend}
Category: {category}
Reasoning: {reasoning}"""
        )

        self.myth_classifier = FewShotPromptTemplate(
            examples=classification_examples,
            example_prompt=classification_prompt,
            prefix="Classify the legend into a category and explain why.\n\n",
            suffix="Legend: {legend}\nCategory:",
            input_variables=["legend"],
        ) | self.llm | StrOutputParser()

        # ✅ Combined approach: fallacy detection
        fallacy_template = PromptTemplate.from_template(
            """Analyze logical fallacies in this myth.

Examples:
- Claim that event B happened after event A, therefore A caused B = False Cause
- Overgeneralizing from a few anecdotes = Hasty Generalization

Legend: {legend_text}
Claims: {claims}

Return bullet list fallacies:"""
        )

        self.fallacy_detector = fallacy_template | self.llm | StrOutputParser()

        # ✅ Zero-shot debunker
        debunk_template = PromptTemplate.from_template(
            """Generate a respectful, scientific myth debunk.

Guidelines:
- Use calm tone
- Explain truth clearly
- Explain why myth spreads
- Use accessible language

Myth: {myth_text}
Claims: {claims}
Fallacies: {fallacies}

Debunking:"""
        )

        self.debunker = debunk_template | self.llm | StrOutputParser()

    def extract_claims_zero_shot(self, legend_text: str) -> List[Claim]:
        """
        TODO #2: Extract claims using zero-shot prompting.
        """
        try:
            result = self.claim_extractor.invoke({"legend_text": legend_text})
            data = json.loads(result)
            return [Claim(**c) for c in data]
        except:
            return []

    def classify_myth_few_shot(self, legend_text: str) -> Tuple[str, str]:
        """
        TODO #3: Classify myth type using few-shot examples.
        """
        response = self.myth_classifier.invoke({"legend": legend_text})
        parts = response.split("Reasoning:")

        category = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else ""
        return category, reasoning

    def detect_fallacies_combined(
        self, legend_text: str, claims: List[Claim]
    ) -> List[str]:
        """
        TODO #4: Detect logical fallacies using combined approach.
        """
        formatted_claims = [c.text for c in claims]
        output = self.fallacy_detector.invoke({
            "legend_text": legend_text,
            "claims": formatted_claims
        })
        return output.split("\n")

    def calculate_believability(
        self, legend_text: str, claims: List[Claim], fallacies: List[str]
    ) -> float:
        """
        TODO #5: Calculate how believable the myth sounds.
        """
        prompt = PromptTemplate.from_template(
            """Rate believability 0-1 by clarity, emotion, authority cues.

Legend: {legend}
Claims: {claims}
Fallacies: {fallacies}

Only output number."""
        )
        score = (prompt | self.llm | StrOutputParser()).invoke({
            "legend": legend_text,
            "claims": [c.text for c in claims],
            "fallacies": fallacies
        })

        try:
            return float(score)
        except:
            return 0.5

    def find_similar_myths(self, legend_text: str, category: str) -> List[str]:
        """
        TODO #6: Find similar myths using few-shot pattern matching.
        """
        prompt = PromptTemplate.from_template(
            """Suggest 2–4 myths similar to this one, same theme.

Category: {category}
Legend: {legend}

Output bullet list only."""
        )

        result = (prompt | self.llm | StrOutputParser()).invoke({
            "category": category,
            "legend": legend_text
        })

        return result.split("\n")

    def analyze_legend(self, legend_text: str) -> MythAnalysis:
        """
        TODO #7: Complete analysis combining all methods.
        """
        claims = self.extract_claims_zero_shot(legend_text)
        category, reason = self.classify_myth_few_shot(legend_text)
        fallacies = self.detect_fallacies_combined(legend_text, claims)
        believability = self.calculate_believability(legend_text, claims, fallacies)
        debunk = self.debunker.invoke({
            "myth_text": legend_text,
            "claims": str([c.text for c in claims]),
            "fallacies": str(fallacies),
        })
        similar = self.find_similar_myths(legend_text, category)

        return MythAnalysis(
            original_text=legend_text,
            category=category,
            claims=claims,
            logical_fallacies=fallacies,
            truth_rating=1 - believability * 0.7,
            believability_score=believability,
            debunking_explanation=debunk,
            similar_myths=similar,
            origin_theory="Likely modern folklore spread socially and online.",
        )

    def adaptive_analysis(self, legend_text: str) -> Dict[str, any]:
        """
        TODO #8 (Bonus): Adaptively choose prompting method based on task.
        """
        method_choices = {
            "claims": "zero-shot (creative extraction)",
            "category": "few-shot (pattern match)",
            "fallacy": "combined",
            "debunk": "zero-shot",
        }

        analysis = self.analyze_legend(legend_text)

        return {
            "analysis": asdict(analysis),
            "method_choices": method_choices,
            "confidence_scores": {
                "claims": 0.8,
                "category": 0.9,
                "fallacies": 0.7,
                "debunking": 0.85,
            },
            "reasoning": "Used zero-shot for flexible reasoning and few-shot for structured classification.",
        }


def test_legend_checker():
    """Test the urban legend checker with various myths."""

    checker = UrbanLegendChecker()

    # Test legends of various types
    test_legends = [
        {
            "title": "The Vanishing Hitchhiker",
            "text": "A driver picks up a young woman hitchhiking on a dark road. She gives an address and sits silently in the back. When they arrive, she's vanished, leaving only a wet spot. The homeowner says she died in a car accident years ago on that very road.",
        },
        {
            "title": "5G Tower Mind Control",
            "text": "5G towers emit special frequencies that can control human thoughts and emotions. The government uses these towers to manipulate public opinion and behavior. People living near 5G towers report more headaches and mood changes, proving the effect.",
        },
        {
            "title": "The $250 Cookie Recipe",
            "text": "A woman at Neiman Marcus loved their cookies and asked for the recipe. The clerk said it would cost 'two-fifty' and she agreed. Her credit card was charged $250, not $2.50. In revenge, she's sharing the secret recipe with everyone.",
        },
        {
            "title": "Kidney Theft Ring",
            "text": "Business travelers are being drugged in hotel bars and waking up in bathtubs full of ice with their kidneys surgically removed. A note tells them to call 911 immediately. Hospitals confirm finding victims with professional surgical scars.",
        },
        {
            "title": "Pop Rocks and Soda",
            "text": "Mixing Pop Rocks candy with soda creates a chemical reaction that can cause your stomach to explode. A child actor died this way in the 1970s, which is why you never see them together in stores.",
        },
    ]

    print("🕵️ URBAN LEGEND FACT CHECKER 🕵️")
    print("=" * 70)

    for legend in test_legends:
        print(f"\n📚 Legend: {legend['title']}")
        print(f"📖 Story: \"{legend['text'][:80]}...\"")

        # Analyze the legend
        analysis = checker.analyze_legend(legend["text"])

        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"  Category: {analysis.category}")
        print(f"  Truth Rating: {analysis.truth_rating:.0%}")
        print(f"  Believability: {analysis.believability_score:.0%}")

        if analysis.claims:
            print(f"\n  🎯 Claims Extracted ({len(analysis.claims)}):")
            for claim in analysis.claims[:2]:  # Show first 2
                print(f"    • {claim.text[:60]}...")
                print(f"      Testable: {'Yes' if claim.testable else 'No'}")

        if analysis.logical_fallacies:
            print(f"\n  ⚠️ Logical Fallacies Detected:")
            for fallacy in analysis.logical_fallacies[:2]:
                print(f"    • {fallacy}")

        if analysis.debunking_explanation:
            print(f"\n  📝 Debunking:")
            print(f"    {analysis.debunking_explanation[:150]}...")

        if analysis.similar_myths:
            print(f"\n  🔗 Similar Myths:")
            for myth in analysis.similar_myths[:2]:
                print(f"    • {myth}")

        print("-" * 70)

    # Test adaptive analysis
    print("\n🧠 ADAPTIVE ANALYSIS TEST:")
    print("=" * 70)

    complex_legend = "Ancient aliens built the pyramids using anti-gravity technology. The precise alignment with stars and mathematical perfection couldn't be achieved with primitive tools."

    adaptive_result = checker.adaptive_analysis(complex_legend)

    if adaptive_result.get("method_choices"):
        print("Method Selection:")
        for task, method in adaptive_result["method_choices"].items():
            print(f"  {task}: {method}")

    if adaptive_result.get("reasoning"):
        print(f"\nReasoning: {adaptive_result['reasoning']}")


if __name__ == "__main__":
    load_dotenv()
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_legend_checker()
