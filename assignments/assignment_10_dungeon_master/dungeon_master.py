"""
Assignment 10: AI Dungeon Master
The Ultimate Challenge - Master all prompting techniques to run a D&D game

Your mission: Become the ultimate AI Dungeon Master by seamlessly combining
all prompting techniques to create epic adventures!
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


class QuestType(Enum):
    RESCUE = "rescue"
    FETCH = "fetch"
    INVESTIGATE = "investigate"
    COMBAT = "combat"
    DIPLOMACY = "diplomacy"
    EXPLORATION = "exploration"


@dataclass
class Character:
    name: str
    class_type: str
    level: int
    hit_points: int
    abilities: List[str]
    inventory: List[str]
    personality: str


@dataclass
class NPC:
    name: str
    role: str
    personality: str
    motivation: str
    dialogue_style: str
    secrets: List[str]


@dataclass
class Quest:
    title: str
    description: str
    objectives: List[str]
    rewards: List[str]
    difficulty: int
    quest_type: str


@dataclass
class CombatState:
    participants: List[Character]
    turn_order: List[str]
    environment: str
    special_conditions: List[str]


@dataclass
class WorldState:
    location: str
    time_of_day: str
    weather: str
    active_quests: List[Quest]
    npcs_present: List[NPC]
    recent_events: List[str]
    player_reputation: Dict[str, int]


class DungeonMasterAI:
    """
    AI Dungeon Master using all prompting techniques seamlessly.
    The ultimate test of prompting mastery!
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.story_generator = None  # Zero-shot
        self.npc_manager = None  # Few-shot
        self.combat_resolver = None  # CoT
        self.world_tracker = None  # Combined
        self.world_state = WorldState(
            location="Tavern",
            time_of_day="Evening",
            weather="Clear",
            active_quests=[],
            npcs_present=[],
            recent_events=[],
            player_reputation={},
        )
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up all chains for different DM tasks.

        Create:
        1. Zero-shot story generator for creative scenarios
        2. Few-shot NPC manager with personality examples
        3. CoT combat resolver for rule calculations
        4. Combined world tracker for state management
        """

        # TODO: Zero-shot for creative story generation
        story_template = PromptTemplate.from_template(
            """Create an engaging D&D scenario.

[TODO: Add instructions for:
- Setting description
- Plot hooks
- Atmospheric details
- Player agency]

Context: {context}
Player Action: {action}

Narration:"""
        )

        # TODO: Few-shot for NPC personalities
        npc_examples = [
            {
                "npc_type": "Gruff Innkeeper",
                "dialogue": "'Ale's two copper, room's a silver. No trouble or you're out.'",
                "personality": "Direct, no-nonsense, secretly kind",
                "quirk": "Always cleaning the same glass",
            },
            {
                "npc_type": "Mysterious Sage",
                "dialogue": "'The answer you seek lies not in what is seen, but what is hidden...'",
                "personality": "Cryptic, wise, slightly mad",
                "quirk": "Speaks in riddles and rhymes",
            },
            # TODO: Add more NPC examples
        ]

        # TODO: CoT for combat calculations
        combat_template = PromptTemplate.from_template(
            """Resolve this D&D combat action step by step.

Action: {action}
Character Stats: {stats}
Target: {target}
Environment: {environment}

Let's calculate the outcome step by step:
Step 1: Check attack roll...
Step 2: Compare to target AC...
Step 3: Calculate damage...
"""
        )

        # TODO: Combined approach for world state tracking
        world_template = PromptTemplate.from_template(
            """Update the world state based on player actions.

[TODO: Combine all techniques for:
- Tracking consequences (CoT)
- Generating reactions (Zero-shot)
- Maintaining consistency (Few-shot)]

Current State: {current_state}
Player Actions: {actions}
Time Passed: {time}

Updated State:"""
        )

        # TODO: Initialize all chains
        pass

    def generate_quest(self, quest_type: QuestType, party_level: int) -> Quest:
        """
        TODO #2: Generate a quest using zero-shot creativity.

        Create unique, engaging quests without examples.
        """

        # TODO: Use zero-shot for creative quest generation

        return Quest(
            title="",
            description="",
            objectives=[],
            rewards=[],
            difficulty=party_level,
            quest_type=quest_type.value,
        )

    def roleplay_npc(self, npc: NPC, player_input: str, context: Dict[str, any]) -> str:
        """
        TODO #3: Roleplay NPC using few-shot personality examples.

        Match personality patterns from examples.
        """

        # TODO: Use few-shot for consistent NPC roleplay

        return ""

    def resolve_combat(
        self,
        action: str,
        attacker: Character,
        target: Character,
        combat_state: CombatState,
    ) -> Dict[str, any]:
        """
        TODO #4: Resolve combat using CoT for rule calculations.

        Step-by-step D&D combat resolution.
        """

        # TODO: Use CoT for accurate combat calculations

        return {"hit": False, "damage": 0, "description": "", "special_effects": []}

    def narrate_scene(
        self, action: str, world_state: WorldState, characters: List[Character]
    ) -> str:
        """
        TODO #5: Narrate scene using zero-shot creativity.

        Generate atmospheric, engaging descriptions.
        """

        # TODO: Create immersive narration

        return ""

    def update_world(self, actions: List[str], time_passed: str) -> WorldState:
        """
        TODO #6: Update world state using ALL techniques.

        Orchestrate all methods for comprehensive world management.
        """

        # TODO: Combine all techniques:
        # - Zero-shot for unexpected consequences
        # - Few-shot for consistent NPC reactions
        # - CoT for logical cause-effect chains

        return self.world_state

    def run_session(
        self, player_actions: List[str], party: List[Character]
    ) -> Dict[str, any]:
        """
        TODO #7: Run a complete game session using all techniques.

        The ultimate test - seamlessly combine everything!
        """

        # TODO: Orchestrate entire game session
        # Switch between techniques as needed
        # Maintain narrative flow
        # Apply rules consistently

        session_log = {
            "narration": [],
            "npc_interactions": [],
            "combat_results": [],
            "quest_updates": [],
            "world_changes": [],
        }

        return session_log


def test_dungeon_master():
    """Test the AI Dungeon Master with a mini adventure."""

    dm = DungeonMasterAI()

    # Create test party
    test_party = [
        Character(
            name="Aldric",
            class_type="Fighter",
            level=3,
            hit_points=28,
            abilities=["Second Wind", "Action Surge"],
            inventory=["Longsword", "Shield", "Healing Potion"],
            personality="Brave but reckless",
        ),
        Character(
            name="Lyra",
            class_type="Wizard",
            level=3,
            hit_points=18,
            abilities=["Fireball", "Shield", "Detect Magic"],
            inventory=["Spellbook", "Crystal Orb", "Scrolls"],
            personality="Cautious and analytical",
        ),
    ]

    # Create test NPCs
    test_npcs = [
        NPC(
            name="Gareth",
            role="Tavern Keeper",
            personality="Gruff but kind",
            motivation="Keep tavern safe",
            dialogue_style="Direct and practical",
            secrets=["Former adventurer", "Has a treasure map"],
        ),
        NPC(
            name="Lady Morwyn",
            role="Noble Patron",
            personality="Aristocratic and mysterious",
            motivation="Find ancient artifact",
            dialogue_style="Formal and cryptic",
            secrets=["Is actually a dragon", "Knows about the prophecy"],
        ),
    ]

    print("🎲 AI DUNGEON MASTER 🎲")
    print("=" * 70)
    print("Welcome to the Realm of Aethermoor!")
    print("-" * 70)

    # Test quest generation (Zero-shot)
    print("\n📜 QUEST GENERATION (Zero-shot):")
    quest = dm.generate_quest(QuestType.RESCUE, party_level=3)
    print(f"Quest: {quest.title}")
    print(f"Description: {quest.description}")
    print(
        f"Objectives: {', '.join(quest.objectives[:2]) if quest.objectives else 'None'}"
    )

    # Test NPC roleplay (Few-shot)
    print("\n🗣️ NPC INTERACTION (Few-shot):")
    player_input = "We're looking for adventure and gold!"
    for npc in test_npcs[:1]:
        response = dm.roleplay_npc(npc, player_input, {"location": "tavern"})
        print(f'{npc.name}: "{response}"')

    # Test combat resolution (CoT)
    print("\n⚔️ COMBAT RESOLUTION (Chain of Thought):")
    combat = CombatState(
        participants=test_party,
        turn_order=[p.name for p in test_party],
        environment="Dark forest clearing",
        special_conditions=["Fog - disadvantage on ranged attacks"],
    )

    combat_result = dm.resolve_combat(
        "Aldric attacks the goblin with his longsword",
        test_party[0],
        Character("Goblin", "Monster", 1, 7, ["Sneak"], ["Dagger"], "Cowardly"),
        combat,
    )

    print(f"Action: Aldric attacks")
    print(f"Result: {'Hit!' if combat_result.get('hit') else 'Miss!'}")
    print(f"Damage: {combat_result.get('damage', 0)}")

    # Test scene narration (Zero-shot)
    print("\n🎭 SCENE NARRATION (Zero-shot):")
    narration = dm.narrate_scene(
        "The party enters the ancient ruins", dm.world_state, test_party
    )
    print(
        f"DM: {narration[:200]}..." if narration else "DM: [Scene description pending]"
    )

    # Test world state update (All techniques)
    print("\n🌍 WORLD STATE UPDATE (All Techniques):")
    player_actions = [
        "Defeated the goblin raiders",
        "Rescued the merchant",
        "Found mysterious artifact",
    ]

    updated_state = dm.update_world(player_actions, "2 hours")
    print(f"Location: {updated_state.location}")
    print(f"Time: {updated_state.time_of_day}")
    print(f"Recent Events: {len(updated_state.recent_events)} recorded")

    # Run mini session
    print("\n🎮 MINI SESSION (All Techniques Combined):")
    print("=" * 70)

    session_actions = [
        "We investigate the strange noises from the cellar",
        "I cast Detect Magic on the mysterious door",
        "We try to negotiate with the creature",
    ]

    session = dm.run_session(session_actions, test_party)

    if session.get("narration"):
        print("Session Highlights:")
        for event in session["narration"][:3]:
            print(f"  • {event}")

    print("\n🏆 Adventure Continues...")
    print("The AI Dungeon Master awaits your next move!")


if __name__ == "__main__":
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
    else:
        test_dungeon_master()
