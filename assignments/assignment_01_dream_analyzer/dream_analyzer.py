"""
Assignment 1: Dream Journal Analyzer
Zero-Shot Prompting - Extract meaning from dreams using only instructions

Your mission: Analyze dream descriptions and extract psychological insights
without any training examples - pure zero-shot magic!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Enums for dream analysis categories
class EmotionType(Enum):
    JOY = "joy"
    FEAR = "fear"
    ANXIETY = "anxiety"
    WONDER = "wonder"
    CONFUSION = "confusion"
    SADNESS = "sadness"
    ANGER = "anger"
    PEACE = "peace"


class DreamTheme(Enum):
    TRANSFORMATION = "transformation"
    PURSUIT = "pursuit/being chased"
    FALLING = "falling"
    FLYING = "flying"
    LOSS = "loss"
    DISCOVERY = "discovery"
    PERFORMANCE = "performance/test"
    RELATIONSHIP = "relationship"
    IDENTITY = "identity"


@dataclass
class DreamSymbol:
    """Represents a symbol found in the dream"""

    symbol: str
    meaning: str
    frequency: int = 1
    significance: float = 0.5  # 0-1 scale


@dataclass
class DreamAnalysis:
    """Complete dream analysis results"""

    symbols: List[DreamSymbol]
    emotions: List[str]
    themes: List[str]
    lucidity_score: float  # 0-10 scale
    psychological_insights: str
    recurring_patterns: List[str]
    dream_type: str  # nightmare, lucid, normal, prophetic


class DreamAnalyzer:
    """
    AI-powered dream journal analyzer using zero-shot prompting.
    Extracts symbols, emotions, and insights from dream descriptions.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize the dream analyzer.

        Args:
            model_name: The LLM model to use
            temperature: Controls creativity (0.0-1.0)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.symbol_chain = None
        self.emotion_chain = None
        self.insight_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for dream analysis components.

        Create THREE chains:
        1. symbol_chain: Extracts symbols and their meanings
        2. emotion_chain: Identifies emotional tones
        3. insight_chain: Generates psychological insights

        Requirements:
        - Use clear, specific instructions
        - Request JSON output format
        - No examples in prompts (zero-shot only!)
        - Handle ambiguous/creative content
        """

        # TODO: Create symbol extraction prompt
        # Hint: Be specific about what constitutes a symbol
        # Include instructions for meaning interpretation
        symbol_template = PromptTemplate.from_template(
            """You are a dream symbol analyzer.

[TODO: Add instructions for:
- What counts as a symbol
- How to interpret meanings
- Output format (JSON)
- Handling abstract concepts]

Dream description: {dream_text}

JSON Output:"""
        )

        # TODO: Create emotion detection prompt
        emotion_template = PromptTemplate.from_template(
            """Analyze the emotional journey in this dream.

[TODO: Add instructions for:
- Types of emotions to detect
- Emotional progression
- Intensity levels
- Output format]

Dream: {dream_text}

Emotions:"""
        )

        # TODO: Create insight generation prompt
        insight_template = PromptTemplate.from_template(
            """Generate psychological insights from this dream.

[TODO: Add instructions for:
- Lucidity score calculation
- Theme identification
- Pattern recognition
- Insight generation]

Dream: {dream_text}
Symbols: {symbols}
Emotions: {emotions}

Analysis:"""
        )

        # TODO: Set up the chains
        self.symbol_chain = None  # Replace with actual chain
        self.emotion_chain = None  # Replace with actual chain
        self.insight_chain = None  # Replace with actual chain

    def extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        """
        TODO #2: Extract symbols and their meanings from dream text.

        Args:
            dream_text: The dream description

        Returns:
            List of DreamSymbol objects with interpretations
        """

        # TODO: Use symbol_chain to extract symbols
        # Parse JSON response and create DreamSymbol objects

        symbols = []

        return symbols

    def analyze_emotions(self, dream_text: str) -> Tuple[List[str], float]:
        """
        TODO #3: Detect emotions and calculate emotional intensity.

        Args:
            dream_text: The dream description

        Returns:
            Tuple of (emotion_list, overall_intensity)
        """

        # TODO: Use emotion_chain to detect emotions
        # Calculate overall emotional intensity (0-10)

        emotions = []
        intensity = 5.0

        return emotions, intensity

    def calculate_lucidity(self, dream_text: str) -> float:
        """
        TODO #4: Calculate lucidity score (awareness level in dream).

        Args:
            dream_text: The dream description

        Returns:
            Lucidity score from 0-10
        """

        # TODO: Create a prompt to assess lucidity indicators
        # Look for: self-awareness, reality checks, control

        lucidity_score = 5.0

        return lucidity_score

    def generate_insights(
        self, dream_text: str, symbols: List[DreamSymbol], emotions: List[str]
    ) -> str:
        """
        TODO #5: Generate psychological insights from dream analysis.

        Args:
            dream_text: The dream description
            symbols: Extracted symbols
            emotions: Detected emotions

        Returns:
            Psychological interpretation and insights
        """

        # TODO: Use insight_chain to generate interpretation
        # Consider symbols, emotions, and themes

        insights = "Dream analysis pending..."

        return insights

    def analyze_dream(self, dream_text: str) -> DreamAnalysis:
        """
        TODO #6: Complete dream analysis pipeline.

        Args:
            dream_text: The dream description to analyze

        Returns:
            Complete DreamAnalysis object with all findings
        """

        # TODO: Implement full analysis pipeline
        # 1. Extract symbols
        # 2. Analyze emotions
        # 3. Calculate lucidity
        # 4. Identify themes
        # 5. Generate insights
        # 6. Create DreamAnalysis object

        analysis = DreamAnalysis(
            symbols=[],
            emotions=[],
            themes=[],
            lucidity_score=0.0,
            psychological_insights="",
            recurring_patterns=[],
            dream_type="normal",
        )

        return analysis

    def compare_dreams(self, dream1: str, dream2: str) -> Dict[str, any]:
        """
        TODO #7 (Bonus): Compare two dreams for similarities and patterns.

        Args:
            dream1: First dream description
            dream2: Second dream description

        Returns:
            Dictionary with similarity scores and shared elements
        """

        # TODO: Implement dream comparison
        # Compare symbols, themes, emotions
        # Calculate similarity score

        comparison = {
            "similarity_score": 0.0,
            "shared_symbols": [],
            "shared_themes": [],
            "pattern_analysis": "",
        }

        return comparison


def test_dream_analyzer():
    """Test the dream analyzer with various dream scenarios."""

    analyzer = DreamAnalyzer()

    # Test dreams with different characteristics
    test_dreams = [
        {
            "title": "The Flying Exam",
            "text": "I was flying over my old school, but suddenly I was in a classroom taking an exam I hadn't studied for. The questions kept changing into pictures of my family. A blue butterfly landed on my paper and whispered the answers.",
        },
        {
            "title": "The Endless Corridor",
            "text": "Walking down a hospital corridor that stretched forever. Every door I opened led to my childhood bedroom, but different versions of it. In one, everything was underwater. In another, the furniture was alive and talking.",
        },
        {
            "title": "The Lucid Garden",
            "text": "I realized I was dreaming when I saw my hands had too many fingers. Decided to create a garden with my thoughts. Purple roses grew instantly, singing a familiar song. I could control the weather by clapping.",
        },
        {
            "title": "The Chase",
            "text": "Something invisible was chasing me through a maze of mirrors. Each reflection showed a different age of myself. When I finally stopped running, the thing chasing me was my own shadow, but it had my mother's voice.",
        },
        {
            "title": "The Time Machine Café",
            "text": "Sitting in a café where each table existed in a different time period. My coffee cup kept refilling with memories instead of coffee. The waiter was my future self, giving me advice I couldn't quite hear.",
        },
    ]

    print("🌙 DREAM JOURNAL ANALYZER 🌙")
    print("=" * 70)

    for dream_data in test_dreams:
        print(f"\n📖 Dream: {dream_data['title']}")
        print(f"💭 Description: \"{dream_data['text'][:80]}...\"")

        # Analyze the dream
        analysis = analyzer.analyze_dream(dream_data["text"])

        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"  Lucidity Score: {analysis.lucidity_score:.1f}/10")
        print(f"  Dream Type: {analysis.dream_type}")

        if analysis.symbols:
            print(f"\n  🔮 Symbols Found ({len(analysis.symbols)}):")
            for symbol in analysis.symbols[:3]:  # Show first 3
                print(f"    • {symbol.symbol}: {symbol.meaning}")

        if analysis.emotions:
            print(f"\n  💝 Emotions Detected:")
            print(f"    {', '.join(analysis.emotions)}")

        if analysis.themes:
            print(f"\n  🎭 Themes:")
            print(f"    {', '.join(analysis.themes)}")

        if analysis.psychological_insights:
            print(f"\n  🧠 Insights:")
            print(f"    {analysis.psychological_insights[:150]}...")

        print("-" * 70)

    # Test dream comparison (bonus)
    print("\n🔄 DREAM COMPARISON TEST:")
    print("=" * 70)

    comparison = analyzer.compare_dreams(
        test_dreams[0]["text"], test_dreams[2]["text"]  # Flying Exam  # Lucid Garden
    )

    print(f"Similarity Score: {comparison.get('similarity_score', 0):.1%}")
    if comparison.get("shared_symbols"):
        print(f"Shared Symbols: {', '.join(comparison['shared_symbols'])}")
    if comparison.get("pattern_analysis"):
        print(f"Pattern Analysis: {comparison['pattern_analysis']}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_dream_analyzer()
