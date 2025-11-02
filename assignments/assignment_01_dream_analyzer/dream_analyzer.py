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
from dotenv import load_dotenv
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

        # ✅ Symbol extraction prompt
        symbol_template = PromptTemplate.from_template(
            """You are a dream symbol analyzer.
Your task is to extract symbolic elements from dreams and identify their psychological meaning.

Instructions:
- A "symbol" is any notable object, creature, environment, action, or transformation in the dream.
- Interpret symbolic meaning based on psychology, mythology, and emotional context.
- If meaning is unclear, infer metaphorically.
- Output ONLY JSON with this structure:
{
  "symbols": [
      {"symbol": "...", "meaning": "...", "frequency": 1, "significance": 0.5}
  ]
}

Dream description: {dream_text}

JSON Output:"""
        )

        # ✅ Emotion detection prompt
        emotion_template = PromptTemplate.from_template(
            """Analyze the emotional tone and emotional journey of this dream.

Instructions:
- Detect emotions like fear, joy, anxiety, wonder, confusion, sadness, anger, peace.
- Consider emotional progression throughout the dream.
- Return a 0-10 emotional intensity score.
- Output only JSON format:
{
  "emotions": ["..."],
  "intensity": 0
}

Dream: {dream_text}

JSON Output:"""
        )

        # ✅ Insight prompt
        insight_template = PromptTemplate.from_template(
            """You are a professional dream psychologist.
Analyze this dream and provide structured psychological insights.

Instructions:
- Rate lucidity (0-10): awareness, control, realization of dream state.
- Identify dream themes (transformation, pursuit, identity, loss, etc.)
- Detect recurring patterns.
- Determine dream type (nightmare, lucid, normal, prophetic tone)
- Provide a short psychological insight paragraph.
- Output JSON only:
{
  "lucidity_score": 0,
  "themes": ["..."],
  "patterns": ["..."],
  "dream_type": "...",
  "insights": "..."
}

Dream: {dream_text}
Symbols: {symbols}
Emotions: {emotions}

JSON Output:"""
        )

        output = StrOutputParser()
        self.symbol_chain = symbol_template | self.llm | output
        self.emotion_chain = emotion_template | self.llm | output
        self.insight_chain = insight_template | self.llm | output

    def extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        """
        TODO #2: Extract symbols and their meanings from dream text.
        """
        symbols = []
        try:
            raw = self.symbol_chain.invoke({"dream_text": dream_text})
            data = json.loads(raw)
            for s in data.get("symbols", []):
                symbols.append(
                    DreamSymbol(
                        symbol=s.get("symbol", ""),
                        meaning=s.get("meaning", ""),
                        frequency=s.get("frequency", 1),
                        significance=float(s.get("significance", 0.5)),
                    )
                )
        except Exception:
            pass

        return symbols

    def analyze_emotions(self, dream_text: str) -> Tuple[List[str], float]:
        """
        TODO #3: Detect emotions and calculate emotional intensity.
        """
        try:
            raw = self.emotion_chain.invoke({"dream_text": dream_text})
            data = json.loads(raw)
            return data.get("emotions", []), float(data.get("intensity", 5.0))
        except Exception:
            return [], 5.0

    def calculate_lucidity(self, dream_text: str) -> float:
        """
        TODO #4: Calculate lucidity score (awareness level in dream).
        """
        keywords = ["realized I was dreaming", "lucid", "controlled", "aware"]
        score = 3
        for k in keywords:
            if k.lower() in dream_text.lower():
                score += 2
        return min(score, 10)

    def generate_insights(
        self, dream_text: str, symbols: List[DreamSymbol], emotions: List[str]
    ) -> str:
        """
        TODO #5: Generate psychological insights from dream analysis.
        """

        try:
            symbols_json = json.dumps([asdict(s) for s in symbols])
            emotions_json = json.dumps(emotions)
            raw = self.insight_chain.invoke(
                {"dream_text": dream_text, "symbols": symbols_json, "emotions": emotions_json}
            )
            data = json.loads(raw)
            return data.get("insights", "")
        except Exception:
            return "Insight generation unsuccessful."

    def analyze_dream(self, dream_text: str) -> DreamAnalysis:
        """
        TODO #6: Complete dream analysis pipeline.
        """

        symbols = self.extract_symbols(dream_text)
        emotions, intensity = self.analyze_emotions(dream_text)
        lucidity = self.calculate_lucidity(dream_text)

        try:
            symbols_json = json.dumps([asdict(s) for s in symbols])
            emotions_json = json.dumps(emotions)
            raw = self.insight_chain.invoke(
                {"dream_text": dream_text, "symbols": symbols_json, "emotions": emotions_json}
            )
            data = json.loads(raw)
        except Exception:
            data = {}

        return DreamAnalysis(
            symbols=symbols,
            emotions=emotions,
            themes=data.get("themes", []),
            lucidity_score=lucidity,
            psychological_insights=data.get("insights", ""),
            recurring_patterns=data.get("patterns", []),
            dream_type=data.get("dream_type", "normal"),
        )

    def compare_dreams(self, dream1: str, dream2: str) -> Dict[str, any]:
        """
        TODO #7 (Bonus): Compare two dreams for similarities and patterns.
        """

        a = self.analyze_dream(dream1)
        b = self.analyze_dream(dream2)

        shared_symbols = list(set([s.symbol for s in a.symbols]) & set([s.symbol for s in b.symbols]))
        shared_themes = list(set(a.themes) & set(b.themes))

        score = (len(shared_symbols) + len(shared_themes)) / max(1, (len(a.symbols) + len(b.symbols)))

        return {
            "similarity_score": score,
            "shared_symbols": shared_symbols,
            "shared_themes": shared_themes,
            "pattern_analysis": "Dreams share symbolic and thematic elements indicating emotional or memory overlap."
        }


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
    load_dotenv()
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_dream_analyzer()
