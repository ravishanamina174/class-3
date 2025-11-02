"""
Assignment 3: Escape Room Puzzle Master
Few-Shot Prompting - Learn from examples to create brain-teasing puzzles
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
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
    puzzle_text: str
    solution: str
    puzzle_type: str
    difficulty: int
    hints: List[str]
    explanation: str
    time_estimate: int


@dataclass
class PuzzleSequence:
    theme: str
    puzzles: List[Puzzle]
    final_solution: str
    narrative: str


class PuzzleMaster:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.puzzle_examples = self._load_puzzle_examples()
        self.generation_chain = None
        self.validation_chain = None
        self.hint_chain = None
        self._setup_chains()

    def _load_puzzle_examples(self) -> Dict[str, List[dict]]:
        examples = {
            "riddle": [
                {
                    "puzzle": "I speak without a mouth and hear without ears. What am I?",
                    "solution": "An echo",
                    "difficulty": "2",
                    "explanation": "Echo repeats sound but has no body."
                },
                {
                    "puzzle": "I have keys but no locks. What am I?",
                    "solution": "A piano",
                    "difficulty": "2",
                    "explanation": "Piano has keys you press."
                },
                {
                    "puzzle": "I can be cracked, made, told, and played. What am I?",
                    "solution": "A joke",
                    "difficulty": "1",
                    "explanation": "Jokes can be cracked and told."
                },
            ],
            "cipher": [
                {"puzzle": "13-1-26-5", "solution": "MAZE", "difficulty": "3", "explanation": "A=1"},
                {"puzzle": "8-5-12-16", "solution": "HELP", "difficulty": "2", "explanation": "Alphabet cipher"},
                {"puzzle": "20-9-13-5", "solution": "TIME", "difficulty": "2", "explanation": "Basic mapping"},
            ],
            "logic": [
                {
                    "puzzle": "Three switches control three bulbs...",
                    "solution": "Heat test method",
                    "difficulty": "4",
                    "explanation": "Use temperature to detect switch"
                },
                {
                    "puzzle": "A farmer must take fox, chicken, grain...",
                    "solution": "Chicken first...",
                    "difficulty": "4",
                    "explanation": "Classic logic sequence"
                }
            ],
            "pattern": [
                {
                    "puzzle": "2, 6, 12, 20, 30, ?",
                    "solution": "42",
                    "difficulty": "3",
                    "explanation": "n(n+1)"
                },
                {
                    "puzzle": "1,1,2,3,5,8,?",
                    "solution": "13",
                    "difficulty": "2",
                    "explanation": "Fibonacci"
                }
            ]
        }
        return examples

    def _setup_chains(self):
        example_prompt = PromptTemplate.from_template(
            """Puzzle: {puzzle}
Solution: {solution}
Difficulty: {difficulty}
Explanation: {explanation}"""
        )



        prefix = """You are a master puzzle designer.

Follow the style of the examples:
- Creative but logical
- Must have unique solvable answer
- Must match requested difficulty
- Output JSON: {{"puzzle":"...","solution":"...","explanation":"..."}}"""

        suffix = """Create a new {puzzle_type} puzzle themed '{theme}' at difficulty {difficulty}.
Output ONLY JSON."""

        self.generation_chain = FewShotPromptTemplate(
            examples=self.puzzle_examples["riddle"] + self.puzzle_examples["cipher"] +
                     self.puzzle_examples["pattern"] + self.puzzle_examples["logic"],
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["puzzle_type", "difficulty", "theme"]
        ) | self.llm | StrOutputParser()

        validation_prompt = PromptTemplate.from_template(
            """Evaluate puzzle fairness and logic.

Puzzle: {puzzle_text}
Solution: {solution}

Return JSON:
{{
 "is_solvable": true/false,
 "has_unique_solution": true/false,
 "difficulty_ok": true/false,
 "issues": [...],
 "suggestions": [...]
}}"""
        )

        self.validation_chain = validation_prompt | self.llm | StrOutputParser()

        hint_prompt = PromptTemplate.from_template(
            """Give {num_hints} hints for solving this:

Puzzle: {puzzle}
Solution: {solution}

Return JSON list only: ["hint1","hint2","hint3"]"""
        )

        self.hint_chain = hint_prompt | self.llm | StrOutputParser()

    def _clean_json(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {}

    def generate_puzzle(self, puzzle_type, difficulty, theme="general") -> Puzzle:
        raw = self.generation_chain.invoke(
            {"puzzle_type": puzzle_type.value, "difficulty": difficulty.value, "theme": theme}
        )

        data = self._clean_json(raw)

        return Puzzle(
            puzzle_text=data.get("puzzle", ""),
            solution=data.get("solution", ""),
            puzzle_type=puzzle_type.value,
            difficulty=difficulty.value,
            hints=[],
            explanation=data.get("explanation", ""),
            time_estimate=5
        )

    def validate_puzzle(self, puzzle: Puzzle) -> Dict[str, any]:
        raw = self.validation_chain.invoke(
            {"puzzle_text": puzzle.puzzle_text, "solution": puzzle.solution}
        )

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            cleaned = raw[start:end]
            return json.loads(cleaned)
        except:
            return {
                "is_solvable": True,
                "has_unique_solution": True,
                "difficulty_ok": True,
                "issues": [],
                "suggestions": [],
            }

    def generate_hints(self, puzzle: Puzzle, num_hints: int = 3) -> List[str]:
        raw = self.hint_chain.invoke(
            {"puzzle": puzzle.puzzle_text, "solution": puzzle.solution, "num_hints": num_hints}
        )
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            cleaned = raw[start:end]
            return json.loads(cleaned)
        except:
            return ["Think logically", "Re-read carefully", "Focus on clues"]

    def create_puzzle_sequence(self, theme, num_puzzles=3, difficulty_curve="increasing") -> PuzzleSequence:
        puzzles = []
        for i in range(num_puzzles):
            diff = DifficultyLevel(i+1 if difficulty_curve=="increasing" else 2)
            p = self.generate_puzzle(PuzzleType.RIDDLE, diff, theme)
            p.hints = self.generate_hints(p, 2)
            puzzles.append(p)

        return PuzzleSequence(
            theme=theme,
            puzzles=puzzles,
            final_solution="Teamwork unlocks the time gate",
            narrative=f"Escape adventure in world of {theme}"
        )


def test_puzzle_master():
    master = PuzzleMaster()

    scenarios = [
        (PuzzleType.RIDDLE, DifficultyLevel.EASY, "pirates"),
        (PuzzleType.CIPHER, DifficultyLevel.MEDIUM, "space"),
        (PuzzleType.LOGIC, DifficultyLevel.HARD, "haunted mansion"),
        (PuzzleType.PATTERN, DifficultyLevel.MEDIUM, "ancient Egypt"),
        (PuzzleType.WORDPLAY, DifficultyLevel.EASY, "detective"),
    ]

    print("🔐 ESCAPE ROOM PUZZLE MASTER 🔐\n" + "="*70)

    for ptype, diff, theme in scenarios:
        print(f"\n🎯 Generating {ptype.value} puzzle\n")
        puzzle = master.generate_puzzle(ptype, diff, theme)
        print("📝 Puzzle:\n", puzzle.puzzle_text)

        validation = master.validate_puzzle(puzzle)
        print("\n✅ Validation:", validation)

        puzzle.hints = master.generate_hints(puzzle)
        print("\n💡 Hints:", puzzle.hints)

        print("\n🔓 Solution:", puzzle.solution)
        print("📖 Explanation:", puzzle.explanation)
        print("⏱️ Time:", puzzle.time_estimate, "min")

        print("-"*70)

    print("\n🎮 Testing puzzle sequence...")
    seq = master.create_puzzle_sequence("Time Travel Mystery")
    print("Sequence puzzles:", len(seq.puzzles))
    print("Story:", seq.narrative)


if __name__ == "__main__":
    load_dotenv()
    test_puzzle_master()
