"""
Assignment 3: Escape Room Puzzle Master
Few-Shot Prompting - Learn from examples to create brain-teasing puzzles

Your mission: Create an AI that learns from puzzle examples to generate
new escape room challenges that are clever, solvable, and fun!
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class PuzzleType(Enum):
    RIDDLE = "riddle"
    CIPHER = "cipher"
    LOGIC = "logic"
    PATTERN = "pattern"
    WORDPLAY = "wordplay"
    VISUAL = "visual"


class DifficultyLevel(Enum):
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class Puzzle:
    """Represents an escape room puzzle"""

    puzzle_text: str
    solution: str
    puzzle_type: str
    difficulty: int
    hints: List[str]
    explanation: str
    time_estimate: int  # minutes


@dataclass
class PuzzleSequence:
    """A series of interconnected puzzles"""

    theme: str
    puzzles: List[Puzzle]
    final_solution: str
    narrative: str


class PuzzleMaster:
    """
    AI-powered escape room puzzle generator using few-shot prompting.
    Learns from examples to create engaging, solvable puzzles.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the puzzle master.

        Args:
            model_name: The LLM model to use
            temperature: Controls creativity (higher = more creative)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.puzzle_examples = self._load_puzzle_examples()
        self.generation_chain = None
        self.validation_chain = None
        self.hint_chain = None
        self._setup_chains()

    def _load_puzzle_examples(self) -> Dict[str, List[dict]]:
        """
        TODO #1: Create example puzzles for few-shot learning.

        Create examples for each puzzle type with consistent format.
        Include puzzle, solution, explanation, and metadata.

        Returns:
            Dictionary mapping puzzle types to example lists
        """

        examples = {
            "riddle": [
                {
                    "puzzle": "I speak without a mouth and hear without ears. I have no body, but come alive with wind. What am I?",
                    "solution": "An echo",
                    "difficulty": "2",
                    "explanation": "An echo repeats sounds (speaks) and responds to sounds (hears) but has no physical form.",
                },
                # TODO: Add 2-3 more riddle examples
            ],
            "cipher": [
                {
                    "puzzle": "Decode: 13-1-26-5",
                    "solution": "MAZE (M=13, A=1, Z=26, E=5)",
                    "difficulty": "3",
                    "explanation": "Simple substitution cipher using alphabetical position numbers.",
                },
                # TODO: Add 2-3 more cipher examples
            ],
            "logic": [
                {
                    "puzzle": "Three switches control three light bulbs in another room. You can only enter the room once. How do you determine which switch controls which bulb?",
                    "solution": "Turn on first switch for 10 minutes, then turn it off. Turn on second switch and enter room. Hot unlit bulb = first switch, lit bulb = second switch, cold unlit = third switch.",
                    "difficulty": "4",
                    "explanation": "Uses the property that incandescent bulbs generate heat when on.",
                },
                # TODO: Add 2-3 more logic examples
            ],
            "pattern": [
                {
                    "puzzle": "Complete the sequence: 2, 6, 12, 20, 30, ?",
                    "solution": "42",
                    "difficulty": "3",
                    "explanation": "Pattern is n*(n+1): 1*2=2, 2*3=6, 3*4=12, 4*5=20, 5*6=30, 6*7=42",
                },
                # TODO: Add 2-3 more pattern examples
            ],
        }

        return examples

    def _setup_chains(self):
        """
        TODO #2: Create few-shot prompt templates for puzzle generation.

        Set up:
        1. generation_chain: Creates new puzzles based on examples
        2. validation_chain: Checks if puzzles are solvable
        3. hint_chain: Generates progressive hints
        """

        # TODO: Create the example prompt template
        example_prompt = PromptTemplate.from_template(
            """Puzzle: {puzzle}
Solution: {solution}
Difficulty: {difficulty}
Explanation: {explanation}"""
        )

        # TODO: Create few-shot template for puzzle generation
        # Include prefix with instructions, examples, and suffix for input

        generation_prefix = """You are a master escape room designer.
        
[TODO: Add instructions for:
- Following the pattern of examples
- Maintaining difficulty level
- Ensuring unique solutions
- Creating engaging narratives]

Here are examples of excellent puzzles:
"""

        generation_suffix = """Now create a new {puzzle_type} puzzle with difficulty {difficulty}.
Theme: {theme}

Generate a puzzle following the exact format of the examples. Output as JSON:"""

        # TODO: Set up the generation chain with FewShotPromptTemplate
        self.generation_chain = None  # Replace with actual chain

        # TODO: Create validation chain with examples of valid/invalid puzzles
        self.validation_chain = None  # Replace with actual chain

        # TODO: Create hint chain with examples of good hint progressions
        self.hint_chain = None  # Replace with actual chain

    def generate_puzzle(
        self,
        puzzle_type: PuzzleType,
        difficulty: DifficultyLevel,
        theme: str = "general",
    ) -> Puzzle:
        """
        TODO #3: Generate a new puzzle using few-shot learning.

        Args:
            puzzle_type: Type of puzzle to generate
            difficulty: Difficulty level (1-5)
            theme: Theme for the puzzle

        Returns:
            Generated Puzzle object
        """

        # TODO: Select appropriate examples based on puzzle type
        # Use generation_chain with selected examples
        # Parse JSON response and create Puzzle object

        puzzle = Puzzle(
            puzzle_text="",
            solution="",
            puzzle_type=puzzle_type.value,
            difficulty=difficulty.value,
            hints=[],
            explanation="",
            time_estimate=5,
        )

        return puzzle

    def validate_puzzle(self, puzzle: Puzzle) -> Dict[str, any]:
        """
        TODO #4: Validate that a puzzle is solvable and fair.

        Args:
            puzzle: The puzzle to validate

        Returns:
            Validation result with solvability score and issues
        """

        # TODO: Use validation_chain with examples of good/bad puzzles
        # Check for: unique solution, logical consistency, appropriate difficulty

        validation = {
            "is_solvable": True,
            "has_unique_solution": True,
            "difficulty_appropriate": True,
            "issues": [],
            "suggestions": [],
        }

        return validation

    def generate_hints(self, puzzle: Puzzle, num_hints: int = 3) -> List[str]:
        """
        TODO #5: Generate progressive hints for a puzzle.

        Args:
            puzzle: The puzzle to generate hints for
            num_hints: Number of hints to generate

        Returns:
            List of hints from subtle to obvious
        """

        # TODO: Use hint_chain with examples of good hint progressions
        # Hints should gradually reveal the solution

        hints = []

        return hints

    def create_puzzle_sequence(
        self, theme: str, num_puzzles: int = 3, difficulty_curve: str = "increasing"
    ) -> PuzzleSequence:
        """
        TODO #6: Create a sequence of interconnected puzzles.

        Args:
            theme: Overall theme for the sequence
            num_puzzles: Number of puzzles in sequence
            difficulty_curve: "increasing", "decreasing", or "varied"

        Returns:
            PuzzleSequence with related puzzles
        """

        # TODO: Generate multiple puzzles that build on each other
        # Solutions from early puzzles can be clues for later ones
        # Maintain narrative coherence

        sequence = PuzzleSequence(
            theme=theme, puzzles=[], final_solution="", narrative=""
        )

        return sequence

    def adapt_difficulty(
        self, puzzle: Puzzle, target_difficulty: DifficultyLevel
    ) -> Puzzle:
        """
        TODO #7 (Bonus): Adapt an existing puzzle to a different difficulty.

        Args:
            puzzle: Original puzzle
            target_difficulty: Desired difficulty level

        Returns:
            Modified puzzle at new difficulty
        """

        # TODO: Use few-shot examples showing difficulty adaptations
        # Easier: add context, simplify language, provide partial solution
        # Harder: add red herrings, require multiple steps, use ambiguity

        adapted_puzzle = puzzle

        return adapted_puzzle


def test_puzzle_master():
    """Test the puzzle master with various scenarios."""

    master = PuzzleMaster()

    # Test different puzzle types and difficulties
    test_scenarios = [
        {
            "type": PuzzleType.RIDDLE,
            "difficulty": DifficultyLevel.EASY,
            "theme": "pirates",
        },
        {
            "type": PuzzleType.CIPHER,
            "difficulty": DifficultyLevel.MEDIUM,
            "theme": "space",
        },
        {
            "type": PuzzleType.LOGIC,
            "difficulty": DifficultyLevel.HARD,
            "theme": "haunted mansion",
        },
        {
            "type": PuzzleType.PATTERN,
            "difficulty": DifficultyLevel.MEDIUM,
            "theme": "ancient Egypt",
        },
        {
            "type": PuzzleType.WORDPLAY,
            "difficulty": DifficultyLevel.EASY,
            "theme": "detective",
        },
    ]

    print("🔐 ESCAPE ROOM PUZZLE MASTER 🔐")
    print("=" * 70)

    for scenario in test_scenarios:
        print(f"\n🎯 Generating {scenario['type'].value} puzzle")
        print(f"   Theme: {scenario['theme']}")
        print(f"   Difficulty: {'⭐' * scenario['difficulty'].value}")

        # Generate puzzle
        puzzle = master.generate_puzzle(
            scenario["type"], scenario["difficulty"], scenario["theme"]
        )

        # Display puzzle
        print(f"\n📝 Puzzle:")
        print(f"   {puzzle.puzzle_text}")

        # Validate puzzle
        validation = master.validate_puzzle(puzzle)
        print(f"\n✅ Validation:")
        print(f"   Solvable: {'Yes' if validation['is_solvable'] else 'No'}")
        print(
            f"   Unique Solution: {'Yes' if validation['has_unique_solution'] else 'No'}"
        )

        # Generate hints
        hints = master.generate_hints(puzzle, num_hints=3)
        if hints:
            print(f"\n💡 Hints:")
            for i, hint in enumerate(hints, 1):
                print(f"   {i}. {hint}")

        # Show solution
        print(f"\n🔓 Solution: {puzzle.solution}")
        print(f"📖 Explanation: {puzzle.explanation}")
        print(f"⏱️ Estimated Time: {puzzle.time_estimate} minutes")

        print("-" * 70)

    # Test puzzle sequence
    print("\n🎮 PUZZLE SEQUENCE TEST:")
    print("=" * 70)

    sequence = master.create_puzzle_sequence(
        theme="Time Travel Mystery", num_puzzles=3, difficulty_curve="increasing"
    )

    print(f"📚 Theme: {sequence.theme}")
    print(f"📖 Narrative: {sequence.narrative}")
    print(f"🎯 Number of Puzzles: {len(sequence.puzzles)}")

    for i, puzzle in enumerate(sequence.puzzles, 1):
        print(f"\n   Puzzle {i}: {puzzle.puzzle_text[:100]}...")
        print(f"   Type: {puzzle.puzzle_type}")
        print(f"   Difficulty: {'⭐' * puzzle.difficulty}")

    if sequence.final_solution:
        print(f"\n🏆 Final Solution: {sequence.final_solution}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_puzzle_master()
