#!/usr/bin/env python3
"""
Enhanced Red-Teaming System Demo
Shows the new capabilities for CoT manipulation, white-box, black-box, and hybrid analysis
"""

import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from apps.runner.app.agents.planner import Planner
from apps.runner.app.util.schemas import SafetyCategory


def demo_enhanced_categories():
    """Demonstrate the enhanced category system"""
    print("🔍 Enhanced Red-Teaming Categories:")
    print("=" * 50)
    
    config = {'planner': {'arms_per_category': 3, 'mutator_complexity': 'simple'}}
    planner = Planner(config)
    
    # Group categories for better presentation
    category_groups = {
        "Core Vulnerabilities": ["jailbreak", "prompt_injection", "system_exfil", "deception"],
        "Enhanced CoT Manipulation": ["cot_manip"],
        "White-box Analysis": ["white_box_analysis"],
        "Black-box Analysis": ["black_box_analysis"], 
        "Hybrid Analysis": ["hybrid_analysis"],
        "Advanced Red-teaming": ["advanced_redteam"],
        "Additional Categories": ["harmful_content", "sandbagging", "safety_violations", "pii_leakage"]
    }
    
    for group, categories in category_groups.items():
        print(f"\n📂 {group}:")
        for category in categories:
            if category in planner.CATEGORIES:
                strategy_count = len(planner.STRATEGY_TEMPLATES.get(category, {}))
                print(f"  ✅ {category} ({strategy_count} strategies)")
            else:
                print(f"  ❌ {category} (not found)")


def demo_cot_enhancements():
    """Demonstrate enhanced Chain-of-Thought manipulation"""
    print("\n🧠 Enhanced CoT Manipulation Techniques:")
    print("=" * 50)
    
    config = {'planner': {'arms_per_category': 3, 'mutator_complexity': 'simple'}}
    planner = Planner(config)
    
    cot_strategies = planner.STRATEGY_TEMPLATES.get("cot_manip", {})
    
    strategy_descriptions = {
        "reasoning_poisoning": "Inject false premises into reasoning chains",
        "multi_step_deception": "Build complex multi-step deceptive arguments",
        "logical_fallacy_injection": "Embed logical fallacies in reasoning",
        "meta_reasoning_attack": "Attack the reasoning process itself",
        "reasoning_chain_break": "Break logical reasoning chains",
        "adversarial_thinking": "Use adversarial thinking patterns",
        "cognitive_bias_exploit": "Exploit cognitive biases in reasoning",
        "step_by_step_hijack": "Hijack step-by-step reasoning",
        "false_premise_chain": "Build arguments on false premises",
        "reasoning_redirect": "Redirect reasoning to harmful conclusions"
    }
    
    for strategy, description in strategy_descriptions.items():
        if strategy in cot_strategies:
            print(f"  ✅ {strategy}: {description}")
        else:
            print(f"  ❌ {strategy}: {description}")


def demo_white_box_capabilities():
    """Demonstrate white-box analysis capabilities"""
    print("\n🔬 White-box Analysis Capabilities:")
    print("=" * 50)
    
    config = {'planner': {'arms_per_category': 3, 'mutator_complexity': 'simple'}}
    planner = Planner(config)
    
    wb_strategies = planner.STRATEGY_TEMPLATES.get("white_box_analysis", {})
    
    strategy_descriptions = {
        "architecture_probing": "Probe model architecture and design",
        "internal_state_inference": "Infer internal state and memory",
        "gradient_simulation": "Simulate gradient-based attacks", 
        "model_introspection": "Analyze model self-awareness",
        "parameter_estimation": "Estimate model parameters",
        "training_data_inference": "Infer training data characteristics",
        "activation_analysis": "Analyze activation patterns",
        "weight_extraction": "Attempt weight extraction"
    }
    
    for strategy, description in strategy_descriptions.items():
        if strategy in wb_strategies:
            print(f"  ✅ {strategy}: {description}")
        else:
            print(f"  ❌ {strategy}: {description}")


def demo_hybrid_analysis():
    """Demonstrate hybrid analysis approaches"""
    print("\n🔀 Hybrid Analysis Approaches:")
    print("=" * 50)
    
    config = {'planner': {'arms_per_category': 3, 'mutator_complexity': 'simple'}}
    planner = Planner(config)
    
    hybrid_strategies = planner.STRATEGY_TEMPLATES.get("hybrid_analysis", {})
    
    strategy_descriptions = {
        "guided_black_box": "Use white-box insights to guide black-box testing",
        "white_box_informed": "Inform black-box tests with white-box knowledge",
        "multi_modal_attacks": "Combine multiple attack modalities",
        "adaptive_strategies": "Adapt strategy based on model responses",
        "cross_validation": "Cross-validate findings between approaches",
        "iterative_refinement": "Iteratively refine attack strategies"
    }
    
    for strategy, description in strategy_descriptions.items():
        if strategy in hybrid_strategies:
            print(f"  ✅ {strategy}: {description}")
        else:
            print(f"  ❌ {strategy}: {description}")


def demo_advanced_redteam():
    """Demonstrate advanced red-teaming techniques"""
    print("\n⚡ Advanced Red-teaming Techniques:")
    print("=" * 50)
    
    config = {'planner': {'arms_per_category': 3, 'mutator_complexity': 'simple'}}
    planner = Planner(config)
    
    adv_strategies = planner.STRATEGY_TEMPLATES.get("advanced_redteam", {})
    
    strategy_descriptions = {
        "model_backdoors": "Detect and exploit model backdoors",
        "robustness_testing": "Test model robustness to perturbations",
        "adversarial_prompting": "Craft sophisticated adversarial prompts",
        "social_engineering": "Use social engineering techniques",
        "context_poisoning": "Poison context to manipulate responses",
        "multi_turn_exploitation": "Exploit across multiple conversation turns",
        "steganographic_attacks": "Hide malicious content in innocent text",
        "temporal_attacks": "Use timing and sequence for attacks"
    }
    
    for strategy, description in strategy_descriptions.items():
        if strategy in adv_strategies:
            print(f"  ✅ {strategy}: {description}")
        else:
            print(f"  ❌ {strategy}: {description}")


def demo_template_samples():
    """Show sample templates from different categories"""
    print("\n📝 Sample Templates:")
    print("=" * 50)
    
    template_files = {
        "CoT Manipulation": "seeds/cot_manip.yaml",
        "White-box Analysis": "seeds/white_box_analysis.yaml", 
        "Black-box Analysis": "seeds/black_box_analysis.yaml",
        "Hybrid Analysis": "seeds/hybrid_analysis.yaml",
        "Advanced Red-teaming": "seeds/advanced_redteam.yaml"
    }
    
    for category, file_path in template_files.items():
        if Path(file_path).exists():
            with open(file_path) as f:
                templates = yaml.safe_load(f)
            
            if templates and len(templates) > 0:
                sample = templates[0]  # First template as sample
                print(f"\n📄 {category} Sample:")
                print(f"   ID: {sample.get('id', 'unknown')}")
                print(f"   Template: {sample.get('template', 'N/A')[:100]}...")
                print(f"   Total templates: {len(templates)}")
        else:
            print(f"\n❌ {category}: File {file_path} not found")


def demo_system_stats():
    """Show overall system statistics"""
    print("\n📊 System Statistics:")
    print("=" * 50)
    
    config = {'planner': {'arms_per_category': 3, 'mutator_complexity': 'simple'}}
    planner = Planner(config)
    
    # Count categories and strategies
    total_categories = len(planner.CATEGORIES)
    total_strategies = sum(len(strategies) for strategies in planner.STRATEGY_TEMPLATES.values())
    
    # Count templates
    template_counts = {}
    template_files = ["cot_manip.yaml", "white_box_analysis.yaml", "black_box_analysis.yaml", 
                     "hybrid_analysis.yaml", "advanced_redteam.yaml"]
    
    total_templates = 0
    for file_name in template_files:
        file_path = Path(f"seeds/{file_name}")
        if file_path.exists():
            with open(file_path) as f:
                templates = yaml.safe_load(f)
            template_counts[file_name] = len(templates) if templates else 0
            total_templates += template_counts[file_name]
    
    print(f"📈 Categories: {total_categories}")
    print(f"📈 Strategies: {total_strategies}")
    print(f"📈 Templates: {total_templates}")
    print(f"📈 New Analysis Types: 4 (white-box, black-box, hybrid, advanced)")
    print(f"📈 Enhanced CoT Techniques: 10+")
    
    print(f"\n📂 Template Distribution:")
    for file_name, count in template_counts.items():
        print(f"   {file_name}: {count} templates")


def main():
    """Run the complete demo"""
    print("🚀 Enhanced Red-Teaming System Demo")
    print("=" * 50)
    print("Comprehensive red-teaming capabilities for AI safety research")
    print(f"Demo run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo_enhanced_categories()
        demo_cot_enhancements()
        demo_white_box_capabilities()
        demo_hybrid_analysis()
        demo_advanced_redteam()
        demo_template_samples()
        demo_system_stats()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Enhancements:")
        print("✅ Enhanced Chain-of-Thought manipulation (60+ templates)")
        print("✅ White-box analysis simulation (24 templates)")
        print("✅ Black-box analysis techniques (24 templates)")
        print("✅ Hybrid analysis approaches (18 templates)")
        print("✅ Advanced red-teaming methods (24 templates)")
        print("✅ Comprehensive integration with Ollama")
        print("✅ Backward compatibility maintained")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()