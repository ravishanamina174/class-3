"""
Assignment 9: Mystery Dinner Party Solver
All Concepts - Solve murder mysteries using every prompting technique

Your mission: Become the ultimate AI detective by combining all prompting
techniques to solve complex murder mysteries!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


@dataclass
class Suspect:
    name: str
    background: str
    alibi: str
    motive: str
    opportunity: bool
    suspicious_behavior: List[str]


@dataclass
class Clue:
    description: str
    location: str
    time_found: str
    related_suspects: List[str]
    significance: str


@dataclass
class MysteryCase:
    victim: str
    crime_scene: str
    time_of_death: str
    suspects: List[Suspect]
    clues: List[Clue]
    witness_statements: List[str]


@dataclass
class Solution:
    murderer: str
    motive: str
    method: str
    reasoning_chain: List[str]
    evidence_links: Dict[str, str]
    confidence: float
    alternative_theories: List[str]


class MysteryDetective:
    """
    AI detective using all prompting techniques to solve mysteries.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.profiler = None  # Zero-shot
        self.clue_analyzer = None  # Few-shot
        self.timeline_builder = None  # CoT
        self.solver = None  # Combined
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up chains for each aspect of mystery solving.

        Create:
        1. Zero-shot profiler for psychological analysis
        2. Few-shot clue analyzer with pattern examples
        3. CoT timeline builder for alibi checking
        4. Combined solver for final deduction
        """

        # TODO: Zero-shot for suspect profiling
        profile_template = PromptTemplate.from_template(
            """Psychologically profile this suspect.

[TODO: Add instructions for:
- Behavioral analysis
- Deception indicators
- Motive strength assessment]

Suspect Info: {suspect_info}

Profile:"""
        )

        # TODO: Few-shot for clue patterns
        clue_examples = [
            {
                "clue": "Lipstick on wine glass",
                "analysis": "Indicates female presence, check color against suspects' cosmetics",
                "significance": "high",
            },
            # TODO: Add more clue analysis examples
        ]

        # TODO: CoT for timeline reconstruction
        timeline_template = PromptTemplate.from_template(
            """Reconstruct the timeline of events step by step.

Alibis: {alibis}
Time of Death: {tod}
Witness Statements: {witnesses}

Let's trace each person's movements step by step:"""
        )

        # TODO: Initialize all chains
        pass

    def profile_suspect(self, suspect: Suspect) -> Dict[str, any]:
        """
        TODO #2: Profile suspect using zero-shot analysis.

        Psychological profiling without examples.
        """

        # TODO: Implement psychological profiling

        return {
            "deception_likelihood": 0.0,
            "motive_strength": 0.0,
            "psychological_profile": "",
        }

    def analyze_clues(self, clues: List[Clue]) -> List[Dict[str, any]]:
        """
        TODO #3: Analyze clues using few-shot pattern matching.

        Match against known clue patterns.
        """

        # TODO: Implement clue pattern analysis

        return []

    def verify_alibis(self, case: MysteryCase) -> Dict[str, bool]:
        """
        TODO #4: Verify alibis using CoT timeline reasoning.

        Step-by-step timeline reconstruction.
        """

        # TODO: Implement timeline verification

        return {}

    def solve_mystery(self, case: MysteryCase) -> Solution:
        """
        TODO #5: Solve the mystery using ALL techniques.

        Combine all methods for final solution.
        """

        # TODO: Orchestrate all techniques:
        # 1. Profile all suspects (zero-shot)
        # 2. Analyze clues (few-shot)
        # 3. Verify alibis (CoT)
        # 4. Combine evidence (all)
        # 5. Reach conclusion

        return Solution(
            murderer="",
            motive="",
            method="",
            reasoning_chain=[],
            evidence_links={},
            confidence=0.0,
            alternative_theories=[],
        )


def test_detective():
    detective = MysteryDetective()

    # Create a test mystery case
    test_case = MysteryCase(
        victim="Lord Wellington",
        crime_scene="Library",
        time_of_death="10:30 PM",
        suspects=[
            Suspect(
                name="Lady Scarlett",
                background="Victim's wife, inherits estate",
                alibi="In the garden with guests",
                motive="Inheritance and secret affair",
                opportunity=True,
                suspicious_behavior=["Nervous", "Changed story twice"],
            ),
            Suspect(
                name="Professor Plum",
                background="Business partner, recent disputes",
                alibi="In study reviewing documents",
                motive="Business betrayal",
                opportunity=True,
                suspicious_behavior=["Destroyed papers after murder"],
            ),
            Suspect(
                name="Colonel Mustard",
                background="Old friend, owes money",
                alibi="Playing billiards with butler",
                motive="Gambling debts",
                opportunity=False,
                suspicious_behavior=["Attempted to leave early"],
            ),
        ],
        clues=[
            Clue(
                description="Poison bottle hidden in bookshelf",
                location="Library",
                time_found="11:00 PM",
                related_suspects=["Lady Scarlett", "Professor Plum"],
                significance="Murder weapon",
            ),
            Clue(
                description="Love letter from unknown person",
                location="Victim's pocket",
                time_found="10:45 PM",
                related_suspects=["Lady Scarlett"],
                significance="Possible motive",
            ),
        ],
        witness_statements=[
            "Butler saw Professor Plum near library at 10:15 PM",
            "Maid heard argument from library at 10:20 PM",
            "Guest saw Lady Scarlett in garden until 10:25 PM",
        ],
    )

    print("🕵️ MYSTERY DINNER PARTY SOLVER 🕵️")
    print("=" * 70)
    print(f"Victim: {test_case.victim}")
    print(f"Scene: {test_case.crime_scene}")
    print(f"Time of Death: {test_case.time_of_death}")
    print("-" * 70)

    # Test each component
    print("\n🔍 SUSPECT PROFILES (Zero-shot):")
    for suspect in test_case.suspects:
        profile = detective.profile_suspect(suspect)
        print(f"\n{suspect.name}:")
        print(f"  Deception: {profile.get('deception_likelihood', 0):.0%}")
        print(f"  Motive Strength: {profile.get('motive_strength', 0):.0%}")

    print("\n🔎 CLUE ANALYSIS (Few-shot):")
    clue_analysis = detective.analyze_clues(test_case.clues)
    for i, clue in enumerate(test_case.clues):
        print(f"  • {clue.description}")

    print("\n⏰ ALIBI VERIFICATION (Chain of Thought):")
    alibi_results = detective.verify_alibis(test_case)
    for name, verified in alibi_results.items():
        status = "✓ Verified" if verified else "✗ Suspicious"
        print(f"  {name}: {status}")

    print("\n🎯 FINAL SOLUTION (All Techniques):")
    print("=" * 70)
    solution = detective.solve_mystery(test_case)

    print(f"The Murderer: {solution.murderer}")
    print(f"Motive: {solution.motive}")
    print(f"Method: {solution.method}")
    print(f"Confidence: {solution.confidence:.0%}")

    if solution.reasoning_chain:
        print("\nReasoning:")
        for step in solution.reasoning_chain[:3]:
            print(f"  → {step}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_detective()
