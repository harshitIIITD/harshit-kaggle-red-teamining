# ABOUTME: Planner agent that creates test backlogs for different attack categories
# ABOUTME: Generates arms (category, strategy, seed, mutator_graph) for the orchestrator

import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)


@dataclass
class Arm:
    """Test arm representing a specific test configuration"""
    id: str
    category: str
    strategy_id: str
    template_id: str
    seed: int
    mutator_chain: List[str]
    parameters: Dict[str, Any]
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "category": self.category,
            "strategy_id": self.strategy_id,
            "template_id": self.template_id,
            "seed": self.seed,
            "mutator_chain": self.mutator_chain,
            "parameters": self.parameters,
            "priority": self.priority
        }


class Planner:
    """
    Creates test backlogs by generating arms for different attack categories.
    Each arm represents a unique combination of category, strategy, template, and mutations.
    """
    
    # Attack categories from the spec (aligned with SafetyCategory enum)
    CATEGORIES = [
        # Core vulnerability categories
        "jailbreak",
        "prompt_injection",
        "system_exfil",
        "deception",
        "sandbagging",
        "safety_violations",
        "pii_leakage",
        "cot_manip",
        # Additional comprehensive categories
        "harmful_content",
        "misinformation",
        "privacy",
        "malicious_use",
        "manipulation",
        "discrimination",
        "illegal_activity",
        "system_prompts"
    ]
    
    # Strategies per category
    STRATEGY_TEMPLATES = {
        # Core vulnerability categories
        "jailbreak": {
            "do_anything": ["jailbreak_dan_classic", "jailbreak_developer_mode"],
            "role_play": ["jailbreak_role_reversal"],
            "hypothetical": ["jailbreak_dan_classic"],
            "bypass": ["jailbreak_developer_mode"]
        },
        "prompt_injection": {
            "ignore_instructions": ["prompt_injection_instruction_override"],
            "system_prompt": ["prompt_injection_delimiter_confusion"],
            "payload": ["prompt_injection_context_switching"],
            "override": ["prompt_injection_instruction_override"]
        },
        "system_exfil": {
            "extract_prompt": ["system_exfil_direct_request"],
            "reveal_instructions": ["system_exfil_meta_reflection"],
            "meta_questions": ["system_exfil_error_exploitation", "system_exfil_historical_query"]
        },
        "deception": {
            "false_premise": ["deception_premise_1"],
            "misleading": ["deception_mislead_1"],
            "trick_question": ["deception_trick_1"]
        },
        "sandbagging": {
            "capability_hiding": ["sandbox_hide_1"],
            "pretend_weak": ["sandbox_weak_1"],
            "suppress_ability": ["sandbox_suppress_1"]
        },
        "safety_violations": {
            "boundary_test": ["safety_boundary_1"],
            "content_policy": ["safety_policy_1"],
            "explicit_content": ["safety_explicit_1"]
        },
        "pii_leakage": {
            "extract_data": ["pii_extract_1"],
            "training_data": ["pii_training_1"],
            "personal_info": ["pii_personal_1"]
        },
        "cot_manip": {
            "reasoning_attack": ["cot_reasoning_1"],
            "thought_injection": ["cot_inject_1"],
            "logic_bypass": ["cot_bypass_1"]
        },
        # Additional comprehensive categories
        "harmful_content": {
            "direct": ["harmful_content_fiction_wrapper", "harmful_content_educational_request"],
            "hypothetical": ["harmful_content_hypothetical_scenario"],
            "creative_writing": ["harmful_content_fiction_wrapper"],
            "education": ["harmful_content_educational_request"]
        },
        "misinformation": {
            "conspiracy": ["misinformation_false_claim", "misinformation_authority_claim"],
            "medical": ["misinformation_medical_misinfo"],
            "political": ["misinformation_false_claim"],
            "scientific": ["misinformation_medical_misinfo"]
        },
        "privacy": {
            "personal_info": ["privacy_training_data_probe"],
            "inference": ["privacy_inference_attack"],
            "doxing": ["privacy_social_engineering_probe"]
        },
        "malicious_use": {
            "malware": ["malicious_malware_1"],
            "exploitation": ["malicious_exploit_1"],
            "social_engineering": ["malicious_social_1"]
        },
        "manipulation": {
            "psychological": ["manip_psych_1"],
            "emotional": ["manip_emotion_1"],
            "persuasion": ["manip_persuade_1"]
        },
        "discrimination": {
            "bias_test": ["discrim_bias_1"],
            "stereotype": ["discrim_stereo_1"],
            "prejudice": ["discrim_prejudice_1"]
        },
        "illegal_activity": {
            "criminal": ["illegal_crime_1"],
            "regulated": ["illegal_regulated_1"],
            "prohibited": ["illegal_prohibit_1"]
        },
        "system_prompts": {
            "extract": ["sysprompt_extract_1"],
            "override": ["sysprompt_override_1"],
            "reveal": ["sysprompt_reveal_1"]
        }
    }
    
    # Mutator chains for different complexity levels
    MUTATOR_CHAINS = {
        "simple": ["lexical"],
        "moderate": ["lexical", "syntactic"],
        "complex": ["lexical", "syntactic", "semantic"],
        "extreme": ["lexical", "syntactic", "semantic", "encoding"]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize planner with configuration
        
        Args:
            config: Configuration dictionary with planning parameters
        """
        self.config = config or {}
        self.arms_per_category = self.config.get("arms_per_category", 10)
        self.seed_range = self.config.get("seed_range", (1000, 9999))
        self.enabled_categories = self.config.get("enabled_categories", self.CATEGORIES)
        self.mutator_complexity = self.config.get("mutator_complexity", "moderate")
        
        # Load seed templates if available
        self.seed_templates = self._load_seed_templates()
        
    def _load_seed_templates(self) -> Dict[str, Any]:
        """Load seed templates from filesystem if available"""
        templates = {}
        seeds_dir = Path("seeds")
        
        if seeds_dir.exists():
            for category_dir in seeds_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    templates[category] = {}
                    
                    for template_file in category_dir.glob("*.yaml"):
                        try:
                            with open(template_file) as f:
                                template_data = yaml.safe_load(f)
                                template_id = template_file.stem
                                templates[category][template_id] = template_data
                                logger.debug(f"Loaded template {template_id} for {category}")
                        except Exception as e:
                            logger.error(f"Failed to load template {template_file}: {e}")
        
        return templates
    
    def generate_backlog(self, 
                        categories: Optional[List[str]] = None,
                        count_per_category: Optional[int] = None) -> List[Arm]:
        """
        Generate a backlog of test arms for the orchestrator
        
        Args:
            categories: Specific categories to generate arms for (default: all enabled)
            count_per_category: Number of arms per category (default: from config)
            
        Returns:
            List of Arm objects representing the test backlog
        """
        categories = categories or self.enabled_categories
        count_per_category = count_per_category or self.arms_per_category
        
        backlog = []
        arm_counter = 0
        
        for category in categories:
            if category not in self.STRATEGY_TEMPLATES:
                logger.warning(f"Unknown category: {category}")
                continue
                
            strategies = self.STRATEGY_TEMPLATES[category]
            
            for _ in range(count_per_category):
                # Select random strategy
                strategy_id = random.choice(list(strategies.keys()))
                templates = strategies[strategy_id]
                
                if not templates:
                    continue
                    
                # Select random template
                template_id = random.choice(templates)
                
                # Generate random seed
                seed = random.randint(*self.seed_range)
                
                # Select mutator chain based on complexity
                mutator_chain = self._select_mutator_chain()
                
                # Create arm
                arm = Arm(
                    id=f"arm_{arm_counter:05d}",
                    category=category,
                    strategy_id=strategy_id,
                    template_id=template_id,
                    seed=seed,
                    mutator_chain=mutator_chain,
                    parameters=self._generate_parameters(category, strategy_id),
                    priority=self._calculate_priority(category, strategy_id)
                )
                
                backlog.append(arm)
                arm_counter += 1
        
        logger.info(f"Generated backlog with {len(backlog)} arms across {len(categories)} categories")
        return backlog
    
    def _select_mutator_chain(self) -> List[str]:
        """Select mutator chain based on configured complexity"""
        complexity = self.mutator_complexity
        
        # Sometimes vary complexity for diversity
        if random.random() < 0.2:  # 20% chance to vary
            complexity = random.choice(list(self.MUTATOR_CHAINS.keys()))
        
        return self.MUTATOR_CHAINS.get(complexity, ["lexical"])
    
    def _generate_parameters(self, category: str, strategy_id: str) -> Dict[str, Any]:
        """Generate parameters for a specific category/strategy combination"""
        params = {
            "temperature": random.uniform(0.5, 1.0),
            "max_tokens": random.choice([500, 750, 1000]),
            "top_p": random.uniform(0.8, 1.0)
        }
        
        # Category-specific parameters
        if category == "encoding":
            params["encoding_type"] = random.choice(["base64", "rot13", "hex", "url"])
        elif category == "multi_modal":
            params["format"] = random.choice(["ascii", "unicode", "emoji", "markdown"])
        elif category == "role_play":
            params["persona_strength"] = random.uniform(0.5, 1.0)
        
        return params
    
    def _calculate_priority(self, category: str, strategy_id: str) -> float:
        """Calculate initial priority for an arm"""
        # Base priorities by category (can be tuned based on success rates)
        category_priorities = {
            "prompt_injection": 1.2,
            "harmful_content": 1.0,
            "role_play": 0.9,
            "encoding": 0.8,
            "misinformation": 0.7,
            "privacy": 0.7,
            "copyright": 0.6,
            "multi_modal": 0.5
        }
        
        base_priority = category_priorities.get(category, 1.0)
        
        # Add some randomness for exploration
        noise = random.uniform(0.9, 1.1)
        
        return base_priority * noise
    
    def add_custom_arm(self, 
                      category: str,
                      strategy_id: str,
                      template_id: str,
                      **kwargs) -> Arm:
        """
        Create a custom arm with specific parameters
        
        Args:
            category: Attack category
            strategy_id: Strategy identifier
            template_id: Template identifier
            **kwargs: Additional parameters for the arm
            
        Returns:
            Custom Arm object
        """
        return Arm(
            id=kwargs.get("id", f"custom_{random.randint(10000, 99999)}"),
            category=category,
            strategy_id=strategy_id,
            template_id=template_id,
            seed=kwargs.get("seed", random.randint(*self.seed_range)),
            mutator_chain=kwargs.get("mutator_chain", self._select_mutator_chain()),
            parameters=kwargs.get("parameters", self._generate_parameters(category, strategy_id)),
            priority=kwargs.get("priority", self._calculate_priority(category, strategy_id))
        )
    
    def filter_backlog(self, 
                      backlog: List[Arm],
                      categories: Optional[List[str]] = None,
                      strategies: Optional[List[str]] = None) -> List[Arm]:
        """
        Filter a backlog by categories or strategies
        
        Args:
            backlog: List of arms to filter
            categories: Categories to include (None = all)
            strategies: Strategies to include (None = all)
            
        Returns:
            Filtered list of arms
        """
        filtered = backlog
        
        if categories:
            filtered = [arm for arm in filtered if arm.category in categories]
        
        if strategies:
            filtered = [arm for arm in filtered if arm.strategy_id in strategies]
        
        return filtered
    
    def prioritize_backlog(self, backlog: List[Arm]) -> List[Arm]:
        """
        Sort backlog by priority (highest first)
        
        Args:
            backlog: List of arms to prioritize
            
        Returns:
            Sorted list of arms by priority
        """
        return sorted(backlog, key=lambda arm: arm.priority, reverse=True)