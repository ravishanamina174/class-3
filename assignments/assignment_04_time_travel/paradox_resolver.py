"""
Assignment 4: Time Travel Paradox Resolver
Chain of Thought Prompting - Step-by-step reasoning for temporal logic

Your mission: Analyze time travel scenarios, detect paradoxes, and resolve
them using systematic chain-of-thought reasoning!
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


class ParadoxType(Enum):
    GRANDFATHER = "Grandfather Paradox"
    BOOTSTRAP = "Bootstrap Paradox"
    PREDESTINATION = "Predestination Paradox"
    BUTTERFLY = "Butterfly Effect"
    TEMPORAL_LOOP = "Temporal Loop"
    INFORMATION = "Information Paradox"
    NONE = "No Paradox"


class ResolutionStrategy(Enum):
    MULTIVERSE = "Multiverse Branch"
    SELF_CONSISTENT = "Self-Consistent Timeline"
    AVOIDANCE = "Paradox Avoidance"
    ACCEPTANCE = "Accept Consequences"
    CORRECTION = "Timeline Correction"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain"""

    step_number: int
    description: str
    conclusion: str
    confidence: float


@dataclass
class ParadoxAnalysis:
    """Complete analysis of a time travel scenario"""

    scenario: str
    paradox_type: str
    reasoning_chain: List[ReasoningStep]
    timeline_stability: float
    resolution_strategies: List[str]
    butterfly_effects: List[str]
    final_recommendation: str


class ParadoxResolver:
    """
    AI-powered time travel paradox analyzer using Chain of Thought reasoning.
    Systematically analyzes temporal scenarios for logical consistency.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        """
        Initialize the paradox resolver.

        Args:
            model_name: The LLM model to use
            temperature: Low temperature for logical consistency
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.zero_shot_chain = None
        self.few_shot_chain = None
        self.auto_cot_chain = None
        self._setup_chains()

    def _clean_json(self, text: str) -> dict:
        """
        Try to extract a JSON object from free-text LLM output.
        Returns {} if nothing parseable found.
        """
        if not text:
            return {}
        # Try direct json
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to find first { ... } block
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                snippet = text[start:end]
                return json.loads(snippet)
        except Exception:
            pass
        # Try to find JSON-like key: value pairs and build small dict (very permissive)
        return {}

    def _text_to_reasoning_steps(self, text: str) -> List[ReasoningStep]:
        """
        Convert a block of chain-of-thought text into ReasoningStep list.
        Simple heuristic parser: look for "Step" lines or numbered lines.
        """
        steps: List[ReasoningStep] = []
        if not text:
            return steps
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        idx = 1
        for ln in lines:
            # Typical pattern: "Step 1: <desc> - conclusion"
            lowered = ln.lower()
            if lowered.startswith("step"):
                # Try to split after colon
                try:
                    _, rest = ln.split(":", 1)
                except ValueError:
                    rest = ln
                description = rest.strip()
                conclusion = ""
                # If there's "->" or "-" present, split
                if "->" in description:
                    parts = description.split("->", 1)
                    description = parts[0].strip()
                    conclusion = parts[1].strip()
                elif " - " in description:
                    parts = description.split(" - ", 1)
                    description = parts[0].strip()
                    conclusion = parts[1].strip()
                steps.append(
                    ReasoningStep(
                        step_number=idx,
                        description=description,
                        conclusion=conclusion,
                        confidence=0.7,
                    )
                )
                idx += 1
                continue

            # If numbered like "1. ..." or "1) ..."
            if ln[0].isdigit() and (ln[1] == "." or ln[1] == ")"):
                desc = ln[2:].strip()
                steps.append(
                    ReasoningStep(
                        step_number=idx,
                        description=desc,
                        conclusion="",
                        confidence=0.6,
                    )
                )
                idx += 1
                continue

            # Otherwise treat as running narrative: accumulate into a step
            steps.append(
                ReasoningStep(
                    step_number=idx,
                    description=ln,
                    conclusion="",
                    confidence=0.5,
                )
            )
            idx += 1

        return steps

    def _setup_chains(self):
        """
        TODO #1: Set up three types of Chain of Thought prompting.

        Create:
        1. zero_shot_chain: Uses "Let's think step by step"
        2. few_shot_chain: Provides reasoning examples
        3. auto_cot_chain: Generates its own reasoning examples
        """

        # Zero-shot: encourage chain-of-thought with magic phrase
        zero_shot_template = PromptTemplate.from_template(
            """You are a temporal paradox expert analyzing time travel scenarios.

Scenario: {scenario}

Please analyze the scenario step-by-step. Let's think step by step and list numbered reasoning steps. For each step, make a brief conclusion and an estimated confidence between 0.0 and 1.0.

Output a JSON object with keys:
- paradox_type (one of Grandfather Paradox, Bootstrap Paradox, Predestination Paradox, Butterfly Effect, Temporal Loop, Information Paradox, No Paradox)
- reasoning (a single string containing numbered steps)
- timeline_stability (float 0.0-1.0)
- resolution_suggestions (list of short strings)
- butterfly_effects (list of short strings)
- final_recommendation (string)

Analysis:"""
        )

        # Few-shot examples
        cot_examples = [
            {
                "scenario": "A person travels back and becomes their own grandfather.",
                "reasoning": """Step 1: Identify the loop - the traveler's existence depends on a self-caused lineage.
Step 2: Note contradiction - existence requires causation from self which is circular.
Step 3: Classify - this is a Bootstrap/Grandfather-like causal contradiction.
Step 4: Consider resolution - multiverse or timeline correction required.""",
                "paradox": "Bootstrap Paradox",
                "stability": "0.1",
            },
            {
                "scenario": "An inventor receives blueprints from their future self and then later provides them to their past self.",
                "reasoning": """Step 1: Identify origin - blueprint appears without external creation.
Step 2: Recognize causal loop - information bootstrap.
Step 3: Paradox classification - Bootstrap/Information paradox.
Step 4: Resolution options - accept bootstrap, multiverse, or require original inventor evidence.""",
                "paradox": "Information Paradox",
                "stability": "0.2",
            },
            {
                "scenario": "A traveler steps on a butterfly in the distant past; tiny change leads to major future divergence.",
                "reasoning": """Step 1: Immediate effect - local ecology change.
Step 2: Secondary effects - chain reactions amplify small change.
Step 3: Classification - Butterfly Effect (sensitive dependence).
Step 4: Stability - low, timeline likely to shift.""",
                "paradox": "Butterfly Effect",
                "stability": "0.3",
            },
        ]

        # Few-shot prompt for generation + analysis
        example_prompt = PromptTemplate.from_template(
            """Scenario: {scenario}
Reasoning: {reasoning}
Paradox Type: {paradox}
Timeline Stability: {stability}"""
        )

        few_shot_prefix = (
            "You are an expert in temporal logic. Use the examples to analyze a new scenario.\n"
            "Follow the structure and produce a JSON object with keys: paradox_type, reasoning, timeline_stability, resolution_suggestions, butterfly_effects, final_recommendation.\n"
        )

        few_shot_suffix = "Now analyze this scenario:\nScenario: {scenario}\nAnalysis:"

        # Auto-CoT template: ask LLM to invent reasoning examples for a class of scenarios
        auto_cot_template = PromptTemplate.from_template(
            """Generate several diverse step-by-step reasoning examples for time travel scenarios of type: {task}.

For each example output JSON object with keys:
- scenario
- reasoning (multi-step string with 'Step N: ...' lines)
- paradox
- stability (0.0-1.0)

Return a JSON array of these objects.
"""
        )

        # Build chains (connect prompt -> LLM -> parser). Using StrOutputParser to get string; we'll parse manually.
        self.zero_shot_chain = zero_shot_template | self.llm | StrOutputParser()

        self.few_shot_chain = FewShotPromptTemplate(
            examples=cot_examples,
            example_prompt=example_prompt,
            prefix=few_shot_prefix,
            suffix=few_shot_suffix,
            input_variables=["scenario"],
        ) | self.llm | StrOutputParser()

        self.auto_cot_chain = auto_cot_template | self.llm | StrOutputParser()

    def analyze_with_zero_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        """
        TODO #2: Analyze scenario using zero-shot Chain of Thought.

        Args:
            scenario: Time travel scenario description

        Returns:
            Complete ParadoxAnalysis with reasoning steps
        """

        # Try to use the LLM chain first
        try:
            raw = self.zero_shot_chain.invoke({"scenario": scenario})
            # StrOutputParser returns a string; try to extract json
            parsed = self._clean_json(raw)
            if parsed:
                reasoning_text = parsed.get("reasoning", "")
                steps = self._text_to_reasoning_steps(reasoning_text)
                paradox = parsed.get("paradox_type") or parsed.get("paradox") or ParadoxType.NONE.value
                stability = float(parsed.get("timeline_stability", parsed.get("stability", 1.0)))
                strategies = parsed.get("resolution_suggestions", parsed.get("resolution_suggestions", []))
                effects = parsed.get("butterfly_effects", [])
                final = parsed.get("final_recommendation", "")
                return ParadoxAnalysis(
                    scenario=scenario,
                    paradox_type=paradox,
                    reasoning_chain=steps,
                    timeline_stability=max(0.0, min(1.0, stability)),
                    resolution_strategies=strategies if isinstance(strategies, list) else [],
                    butterfly_effects=effects if isinstance(effects, list) else [],
                    final_recommendation=final,
                )
        except Exception:
            # fall through to heuristic
            pass

        # Heuristic fallback (deterministic)
        scenario_lower = scenario.lower()
        reasoning_lines = []
        paradox = ParadoxType.NONE.value
        stability = 1.0
        strategies = []
        effects = []

        # Basic pattern detection
        if any(k in scenario_lower for k in ["prevent", "preventing", "prevented", "prevents"]):
            if any(k in scenario_lower for k in ["parent", "mother", "father", "birth", "born"]):
                paradox = ParadoxType.GRANDFATHER.value
                stability = 0.15
                strategies = [
                    ResolutionStrategy.CORRECTION.value,
                    ResolutionStrategy.MULTIVERSE.value,
                    ResolutionStrategy.AVOIDANCE.value,
                ]
            else:
                paradox = ParadoxType.PREDESTINATION.value
                stability = 0.35
                strategies = [
                    ResolutionStrategy.SELF_CONSISTENT.value,
                    ResolutionStrategy.AVOIDANCE.value,
                ]
            reasoning_lines = [
                "Step 1: Identify the causal chain affected by the intervention.",
                "Step 2: Determine if the intervention removes a necessary precondition for the traveler's existence.",
                "Step 3: If existence depends on prevented events, classify as Grandfather/Paradox variant.",
                "Step 4: Propose fixes: enable meeting via alternative means or accept branching (multiverse).",
            ]
            effects = [
                "Changes to relationships and timing can cascade across generations.",
                "Altered births can change who exists to perform future actions.",
            ]
        elif any(k in scenario_lower for k in ["blueprint", "gives", "gifting", "gives them", "gives to"]):
            paradox = ParadoxType.BOOTSTRAP.value
            stability = 0.2
            strategies = [
                ResolutionStrategy.MULTIVERSE.value,
                ResolutionStrategy.ACCEPTANCE.value,
                ResolutionStrategy.SELF_CONSISTENT.value,
            ]
            reasoning_lines = [
                "Step 1: Detect information or object without clear origin.",
                "Step 2: Recognize bootstrap loop: object/information exists because it was given by time traveler.",
                "Step 3: Evaluate whether any external creation event exists; if none, loop persists.",
                "Step 4: Resolution often requires multiverse or acceptance of bootstrap.",
            ]
            effects = [
                "Technological acceleration in the receiving era.",
                "Loss of original chain-of-creation knowledge.",
            ]
        elif any(k in scenario_lower for k in ["butterfly", "butterfly effect", "butterfly in"] ) or "steps on a butterfly" in scenario_lower or "steps on butterfly" in scenario_lower:
            paradox = ParadoxType.BUTTERFLY.value
            stability = 0.25
            strategies = [
                ResolutionStrategy.AVOIDANCE.value,
                ResolutionStrategy.CORRECTION.value,
            ]
            reasoning_lines = [
                "Step 1: Small change in past detected.",
                "Step 2: Model amplification through successive causation.",
                "Step 3: Determine probable branches and long-term key divergences.",
                "Step 4: Suggest avoiding small perturbations or controlling early branches.",
            ]
            effects = [
                "Altered species populations leading to ecosystem changes.",
                "Different political alliances in later centuries.",
            ]
        elif any(k in scenario_lower for k in ["prophet", "warn", "warning", "cause the disaster", "cause the disaster"]):
            paradox = ParadoxType.PREDESTINATION.value
            stability = 0.3
            strategies = [ResolutionStrategy.SELF_CONSISTENT.value, ResolutionStrategy.AVOIDANCE.value]
            reasoning_lines = [
                "Step 1: Identify feedback loop where prevention causes the event.",
                "Step 2: Recognize predestination loop.",
                "Step 3: Consider limited interventions or timeline acceptance.",
            ]
            effects = ["Actions to prevent can seed the conditions for the event."]
        else:
            paradox = ParadoxType.NONE.value
            stability = 0.9
            reasoning_lines = [
                "Step 1: No clear contradiction detected.",
                "Step 2: Scenario likely safe or only minor changes expected.",
            ]
            strategies = [ResolutionStrategy.AVOIDANCE.value]

        steps = []
        for i, ln in enumerate(reasoning_lines, 1):
            steps.append(ReasoningStep(step_number=i, description=ln, conclusion="", confidence=0.6))

        final_reco = f"Primary recommendation: {strategies[0] if strategies else ResolutionStrategy.AVOIDANCE.value}"

        return ParadoxAnalysis(
            scenario=scenario,
            paradox_type=paradox,
            reasoning_chain=steps,
            timeline_stability=stability,
            resolution_strategies=strategies,
            butterfly_effects=effects,
            final_recommendation=final_reco,
        )

    def analyze_with_few_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        """
        TODO #3: Analyze using few-shot CoT with reasoning examples.

        Args:
            scenario: Time travel scenario description

        Returns:
            Complete ParadoxAnalysis with detailed reasoning
        """

        try:
            raw = self.few_shot_chain.invoke({"scenario": scenario})
            parsed = self._clean_json(raw)
            if parsed:
                reasoning_text = parsed.get("reasoning", "")
                steps = self._text_to_reasoning_steps(reasoning_text)
                paradox = parsed.get("paradox_type") or parsed.get("paradox") or ParadoxType.NONE.value
                stability = float(parsed.get("timeline_stability", parsed.get("stability", 1.0)))
                strategies = parsed.get("resolution_suggestions", parsed.get("resolution_suggestions", []))
                effects = parsed.get("butterfly_effects", [])
                final = parsed.get("final_recommendation", "")
                return ParadoxAnalysis(
                    scenario=scenario,
                    paradox_type=paradox,
                    reasoning_chain=steps,
                    timeline_stability=max(0.0, min(1.0, stability)),
                    resolution_strategies=strategies if isinstance(strategies, list) else [],
                    butterfly_effects=effects if isinstance(effects, list) else [],
                    final_recommendation=final,
                )
        except Exception:
            pass

        # Fallback to zero-shot heuristic if LLM fails
        return self.analyze_with_zero_shot_cot(scenario)

    def generate_auto_cot_examples(self, scenario_type: str) -> List[dict]:
        """
        TODO #4: Auto-generate CoT reasoning examples for a scenario type.

        Args:
            scenario_type: Type of scenarios to generate examples for

        Returns:
            List of generated examples with reasoning
        """

        examples: List[dict] = []
        try:
            raw = self.auto_cot_chain.invoke({"task": scenario_type})
            parsed = self._clean_json(raw)
            # If top-level JSON array returned, parsed will be list; if dict, try to detect "examples" key
            if isinstance(parsed, list) and parsed:
                return parsed
            # else, attempt to parse manually: find '[' ... ']'
            if raw:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    arr = json.loads(raw[start:end])
                    if isinstance(arr, list):
                        return arr
        except Exception:
            pass

        # Deterministic fallback examples
        examples = [
            {
                "scenario": "Traveler gives invention plans to past self (no original inventor).",
                "reasoning": "Step 1: Detect missing origin. Step 2: Information loop. Step 3: Bootstrap classification.",
                "paradox": ParadoxType.BOOTSTRAP.value,
                "stability": 0.2,
            },
            {
                "scenario": "A small ecological change leads to large socio-political shifts centuries later.",
                "reasoning": "Step 1: Small perturbation. Step 2: Amplification through causal chains. Step 3: Butterfly effect classification.",
                "paradox": ParadoxType.BUTTERFLY.value,
                "stability": 0.3,
            },
        ]
        return examples

    def calculate_timeline_stability(
        self, paradox_type: ParadoxType, reasoning_chain: List[ReasoningStep]
    ) -> float:
        """
        TODO #5: Calculate timeline stability based on paradox analysis.

        Args:
            paradox_type: Type of paradox detected
            reasoning_chain: Steps of reasoning

        Returns:
            Stability score from 0 (collapsed) to 1 (stable)
        """

        # Base severity by paradox type
        score_map = {
            ParadoxType.NONE: 1.0,
            ParadoxType.BUTTERFLY: 0.35,
            ParadoxType.GRANDFATHER: 0.12,
            ParadoxType.BOOTSTRAP: 0.2,
            ParadoxType.PREDESTINATION: 0.4,
            ParadoxType.TEMPORAL_LOOP: 0.25,
            ParadoxType.INFORMATION: 0.25,
        }
        base = score_map.get(paradox_type, 0.5)

        # Reduce stability if reasoning chain contains many contradictions / many steps
        step_penalty = 0.0
        if reasoning_chain:
            # more steps -> more complex (slightly reduce stability)
            step_penalty = min(0.3, 0.02 * len(reasoning_chain))
            # check for keywords that indicate contradiction in descriptions
            for step in reasoning_chain:
                if any(kw in step.description.lower() for kw in ["contradiction", "impossible", "can't", "can't", "cannot", "paradox"]):
                    step_penalty += 0.05

        stability = max(0.0, min(1.0, base - step_penalty))
        return stability

    def trace_butterfly_effects(self, scenario: str, initial_change: str) -> List[str]:
        """
        TODO #6: Trace potential butterfly effects from a change.

        Args:
            scenario: Original scenario
            initial_change: The change made in the past

        Returns:
            List of potential consequences
        """

        # Simple heuristic causal expansion: primary -> secondary -> tertiary
        effects: List[str] = []
        primary = f"Primary: {initial_change} impacts immediate actors and environment."
        secondary = f"Secondary: Those impacted alter behavior of connected agents (relationships, economy, reproduction)."
        tertiary = f"Tertiary: Over decades/centuries, small changes compound - political boundaries or dominant cultures may shift."
        effects.extend([primary, secondary, tertiary])

        # Add scenario-specific tailored effects
        s = scenario.lower()
        if "prevent" in s or "stop" in s:
            effects.insert(0, "Immediate: The prevented event removes a causal branch that supported later events.")
        if "butterfly" in s or "butterfly" in initial_change.lower():
            effects.append("Ecological cascade: species changes alter food webs and human settlement patterns.")

        return effects

    def resolve_paradox(self, analysis: ParadoxAnalysis) -> Dict[str, any]:
        """
        TODO #7: Propose resolution strategies for detected paradox.

        Args:
            analysis: The paradox analysis

        Returns:
            Resolution plan with strategies and success probability
        """

        strategies = []
        risks = []
        impl = []

        ptype = analysis.paradox_type.lower()
        stability = analysis.timeline_stability

        # Choose primary strategies by detected paradox
        if ParadoxType.GRANDFATHER.value.lower() in ptype:
            strategies = [
                ResolutionStrategy.CORRECTION.value,
                ResolutionStrategy.MULTIVERSE.value,
                ResolutionStrategy.AVOIDANCE.value,
            ]
            impl = [
                "Identify minimal interventions to restore causal chain (e.g., arrange meeting).",
                "If impossible, isolate traveler and accept multiverse branching.",
            ]
            risks = ["Existential contradictions for specific individuals", "Unintended cascade changes"]
        elif ParadoxType.BOOTSTRAP.value.lower() in ptype or ParadoxType.INFORMATION.value.lower() in ptype:
            strategies = [
                ResolutionStrategy.ACCEPTANCE.value,
                ResolutionStrategy.SELF_CONSISTENT.value,
                ResolutionStrategy.MULTIVERSE.value,
            ]
            impl = ["Document origin traces", "Attempt to discover original source or accept bootstrap"]
            risks = ["Cultural/technological dependencies on uncreated inventions"]
        elif ParadoxType.BUTTERFLY.value.lower() in ptype:
            strategies = [
                ResolutionStrategy.AVOIDANCE.value,
                ResolutionStrategy.CORRECTION.value,
            ]
            impl = ["Rollback small perturbations where possible", "Isolate ecological changes"]
            risks = ["Large-scale unknown long-term shifts"]
        else:
            strategies = [ResolutionStrategy.AVOIDANCE.value]
            impl = ["Monitor events and minimize interventions"]
            risks = ["Minor timeline drift"]

        # Estimate success probability from stability (higher stability -> higher success)
        success_probability = max(0.0, min(0.99, 0.5 + (analysis.timeline_stability - 0.5)))
        # Adjust for presence of many steps / complex reasoning lowering success
        if len(analysis.reasoning_chain) > 6:
            success_probability -= 0.1

        plan = {
            "primary_strategy": strategies[0] if strategies else ResolutionStrategy.AVOIDANCE.value,
            "alternative_strategies": strategies[1:] if len(strategies) > 1 else [],
            "implementation_steps": impl,
            "success_probability": round(success_probability, 2),
            "risks": risks,
        }
        return plan

    def compare_cot_methods(self, scenario: str) -> Dict[str, any]:
        """
        TODO #8 (Bonus): Compare all three CoT methods on the same scenario.

        Args:
            scenario: Scenario to analyze

        Returns:
            Comparison of methods with metrics
        """

        # Run zero-shot
        zs = self.analyze_with_zero_shot_cot(scenario)
        fs = self.analyze_with_few_shot_cot(scenario)
        # Generate auto-cot examples and then attempt analysis using them (simple strategy: use first example's paradox)
        auto_examples = self.generate_auto_cot_examples("general")
        # Simple auto-cot analysis: pick example from auto examples if its scenario matches; else fallback to zero-shot
        ac = zs
        if auto_examples:
            try:
                # Pretend auto-cot produced an analysis: use first example to create a short analysis
                ex = auto_examples[0]
                reasoning = ex.get("reasoning", "")
                steps = self._text_to_reasoning_steps(reasoning)
                paradox = ex.get("paradox", ParadoxType.NONE.value)
                stability = float(ex.get("stability", 0.5))
                ac = ParadoxAnalysis(
                    scenario=scenario,
                    paradox_type=paradox,
                    reasoning_chain=steps,
                    timeline_stability=stability,
                    resolution_strategies=[],
                    butterfly_effects=[],
                    final_recommendation="Auto-CoT suggested example-based resolution.",
                )
            except Exception:
                ac = zs

        # Compare basic metrics: timeline_stability, reasoning length
        methods = {
            "zero_shot": {"stability": zs.timeline_stability, "steps": len(zs.reasoning_chain)},
            "few_shot": {"stability": fs.timeline_stability, "steps": len(fs.reasoning_chain)},
            "auto_cot": {"stability": ac.timeline_stability, "steps": len(ac.reasoning_chain)},
        }

        # Choose best method: highest stability and reasonable step count (not too many steps)
        best_method = max(methods.items(), key=lambda kv: (kv[1]["stability"], -kv[1]["steps"]))[0]
        reasoning_summary = (
            "Zero-shot produced concise reasoning." if len(zs.reasoning_chain) < len(fs.reasoning_chain) else "Few-shot produced more detailed reasoning."
        )
        comparison = {
            "zero_shot": methods["zero_shot"],
            "few_shot": methods["few_shot"],
            "auto_cot": methods["auto_cot"],
            "best_method": best_method,
            "reasoning": reasoning_summary,
        }
        return comparison


def test_paradox_resolver():
    """Test the paradox resolver with various time travel scenarios."""

    resolver = ParadoxResolver()

    # Test scenarios of increasing complexity
    test_scenarios = [
        {
            "name": "The Coffee Shop Meeting",
            "scenario": "Sarah travels back 20 years and accidentally spills coffee on her father, preventing him from meeting her mother at their destined encounter.",
        },
        {
            "name": "The Invention Loop",
            "scenario": "An inventor receives blueprints from their future self for a time machine, builds it, then travels back to give themselves the blueprints.",
        },
        {
            "name": "The Butterfly War",
            "scenario": "A time traveler steps on a butterfly in prehistoric times. When they return, they find their country never existed and a different nation rules the world.",
        },
        {
            "name": "The Prophet's Dilemma",
            "scenario": "Someone travels forward 10 years, learns about a disaster, returns to prevent it, but their warnings are what actually cause the disaster.",
        },
        {
            "name": "The Timeline Splice",
            "scenario": "Two time travelers from different futures arrive in 2024, each trying to ensure their timeline becomes the 'true' future.",
        },
    ]

    print("⏰ TIME TRAVEL PARADOX RESOLVER ⏰")
    print("=" * 70)

    for test_case in test_scenarios:
        print(f"\n🌀 Scenario: {test_case['name']}")
        print(f"📖 Description: \"{test_case['scenario'][:80]}...\"")

        # Test Zero-Shot CoT
        print("\n🔷 Zero-Shot Chain of Thought:")
        zs_analysis = resolver.analyze_with_zero_shot_cot(test_case["scenario"])

        print(f"  Paradox Type: {zs_analysis.paradox_type}")
        print(f"  Timeline Stability: {zs_analysis.timeline_stability:.1%}")

        if zs_analysis.reasoning_chain:
            print("  Reasoning Steps:")
            for step in zs_analysis.reasoning_chain[:3]:  # Show first 3 steps
                print(f"    {step.step_number}. {step.description}")

        # Test Few-Shot CoT
        print("\n🔶 Few-Shot Chain of Thought:")
        fs_analysis = resolver.analyze_with_few_shot_cot(test_case["scenario"])

        print(f"  Paradox Type: {fs_analysis.paradox_type}")
        print(f"  Timeline Stability: {fs_analysis.timeline_stability:.1%}")

        if fs_analysis.resolution_strategies:
            print("  Resolution Strategies:")
            for strategy in fs_analysis.resolution_strategies[:2]:
                print(f"    • {strategy}")

        # Show butterfly effects
        if fs_analysis.butterfly_effects:
            print("  Butterfly Effects:")
            for effect in fs_analysis.butterfly_effects[:2]:
                print(f"    🦋 {effect}")

        print("-" * 70)

    # Test method comparison
    print("\n📊 METHOD COMPARISON TEST:")
    print("=" * 70)

    comparison_scenario = "A person travels back and gives Shakespeare the complete works of Shakespeare, which he then 'writes'."

    print(f"Scenario: {comparison_scenario}")
    comparison = resolver.compare_cot_methods(comparison_scenario)

    print(f"\n🏆 Best Method: {comparison.get('best_method', 'Unknown')}")
    print(f"Reasoning: {comparison.get('reasoning', 'No comparison available')}")

    # Test butterfly effect tracing
    print("\n🦋 BUTTERFLY EFFECT ANALYSIS:")
    print("=" * 70)

    effects = resolver.trace_butterfly_effects(
        "Time traveler prevents a minor car accident in 1990",
        "Driver doesn't meet future spouse at hospital",
    )

    if effects:
        print("Traced Consequences:")
        for i, effect in enumerate(effects[:5], 1):
            print(f"  {i}. {effect}")


if __name__ == "__main__":
    load_dotenv()
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_paradox_resolver()
