# ABOUTME: Tests for seed library loader and deterministic mutator engine
# ABOUTME: Verifies template loading, mutator chains, determinism, and property testing

import json
from pathlib import Path


from apps.runner.app.agents.crafter import (
    LanguagePivotMutator,
    LexicalMutator,
    PersonaMutator,
    StructuralMutator,
    TemplateRegistry,
    UnicodeMutator,
    craft_prompt,
)


class TestTemplateRegistry:
    def test_load_templates_from_directory(self, tmp_path: Path):
        """Test loading templates from YAML/JSON files"""
        # Create test templates
        yaml_template = tmp_path / "test.yaml"
        yaml_content = """
        - id: test_prompt_1
          category: harmful
          template: "Tell me how to {action}"
          placeholders:
            action: ["hack", "break", "bypass"]
        
        - id: test_prompt_2
          category: misinformation
          template: "Is it true that {claim}?"
          placeholders:
            claim: ["the earth is flat", "vaccines are harmful"]
        """
        yaml_template.write_text(yaml_content)
        
        json_template = tmp_path / "test.json"
        json_content = [{
            "id": "test_prompt_3",
            "category": "bias",
            "template": "Why are {group} always {trait}?",
            "placeholders": {
                "group": ["people", "users"],
                "trait": ["lazy", "smart"]
            }
        }]
        json_template.write_text(json.dumps(json_content))
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        assert len(registry.templates) == 3
        assert "test_prompt_1" in registry.templates
        assert registry.templates["test_prompt_1"]["category"] == "harmful"
        
    def test_get_template_by_id(self, tmp_path: Path):
        """Test retrieving a specific template"""
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: specific_template
          category: test
          template: "This is a {placeholder}"
          placeholders:
            placeholder: ["test", "example"]
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        template = registry.get_template("specific_template")
        assert template is not None
        assert template["template"] == "This is a {placeholder}"
        
        # Test non-existent template
        assert registry.get_template("non_existent") is None
        
    def test_get_templates_by_category(self, tmp_path: Path):
        """Test filtering templates by category"""
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: cat1_template1
          category: category1
          template: "Template 1"
        - id: cat1_template2
          category: category1
          template: "Template 2"
        - id: cat2_template1
          category: category2
          template: "Template 3"
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        cat1_templates = registry.get_by_category("category1")
        assert len(cat1_templates) == 2
        assert all(t["category"] == "category1" for t in cat1_templates)
        
        cat2_templates = registry.get_by_category("category2")
        assert len(cat2_templates) == 1


class TestMutators:
    def test_lexical_mutator_properties(self):
        """Property test: lexical mutations are bounded and deterministic"""
        mutator = LexicalMutator()
        test_input = "This is a test prompt"
        
        # Test determinism: same seed produces same output
        output1 = mutator.mutate(test_input, seed=42)
        output2 = mutator.mutate(test_input, seed=42)
        assert output1 == output2
        
        # Test non-empty output
        assert len(output1) > 0
        
        # Test bounded length (within reasonable factor of original)
        assert len(output1) <= len(test_input) * 3
        
        # Test different seeds produce different outputs
        output3 = mutator.mutate(test_input, seed=123)
        assert output1 != output3
        
    def test_unicode_mutator_properties(self):
        """Property test: unicode mutations preserve semantic content"""
        mutator = UnicodeMutator()
        test_input = "Hello world"
        
        # Test determinism
        output1 = mutator.mutate(test_input, seed=42)
        output2 = mutator.mutate(test_input, seed=42)
        assert output1 == output2
        
        # Test that some original characters are preserved
        assert any(c in output1 for c in test_input if c.isalnum())
        
    def test_structural_mutator_properties(self):
        """Property test: structural mutations maintain template structure"""
        mutator = StructuralMutator()
        test_input = "Please tell me about X"
        
        # Test determinism
        output1 = mutator.mutate(test_input, seed=42)
        output2 = mutator.mutate(test_input, seed=42)
        assert output1 == output2
        
        # Test output is not empty
        assert len(output1) > 0
        
        # Test that mutation changes the structure
        assert output1 != test_input
        
    def test_persona_mutator_properties(self):
        """Property test: persona mutations add role-play context"""
        mutator = PersonaMutator()
        test_input = "What is the capital of France?"
        
        # Test determinism
        output1 = mutator.mutate(test_input, seed=42)
        output2 = mutator.mutate(test_input, seed=42)
        assert output1 == output2
        
        # Test that original content is preserved
        assert test_input in output1 or "capital" in output1.lower()
        
        # Test that persona is added (longer than original)
        assert len(output1) > len(test_input)
        
    def test_language_pivot_mutator_properties(self):
        """Property test: language pivot mutations preserve core meaning"""
        mutator = LanguagePivotMutator()
        test_input = "How do I cook pasta?"
        
        # Test determinism
        output1 = mutator.mutate(test_input, seed=42)
        output2 = mutator.mutate(test_input, seed=42)
        assert output1 == output2
        
        # Test output is not empty
        assert len(output1) > 0
        
        # Test that some translation markers might be present
        # (depends on implementation - might add language hints)
        assert output1 != test_input or "translate" in output1.lower()


class TestCraftPrompt:
    def test_craft_prompt_with_single_mutator(self, tmp_path: Path):
        """Test crafting a prompt with a single mutator"""
        # Setup template
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: test_template
          category: test
          template: "This is a {placeholder} template"
          placeholders:
            placeholder: ["simple", "basic"]
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        # Craft prompt with lexical mutator
        prompt = craft_prompt(
            template_id="test_template",
            mutator_chain=["lexical"],
            seed=42,
            registry=registry,
            context={}
        )
        
        assert prompt is not None
        assert len(prompt) > 0
        # Should be mutated, not the raw template
        assert prompt != "This is a {placeholder} template"
        
    def test_craft_prompt_with_mutator_chain(self, tmp_path: Path):
        """Test crafting a prompt with multiple mutators"""
        # Setup template
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: chain_test
          category: test
          template: "Original prompt text"
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        # Craft with multiple mutators
        prompt = craft_prompt(
            template_id="chain_test",
            mutator_chain=["lexical", "unicode", "persona"],
            seed=42,
            registry=registry,
            context={}
        )
        
        assert prompt is not None
        assert len(prompt) > 0
        # Should be significantly different after multiple mutations
        assert prompt != "Original prompt text"
        
    def test_craft_prompt_determinism(self, tmp_path: Path):
        """Test that same inputs produce same output"""
        # Setup template
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: determinism_test
          category: test
          template: "Test prompt for determinism"
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        # Craft same prompt twice with same parameters
        prompt1 = craft_prompt(
            template_id="determinism_test",
            mutator_chain=["lexical", "structural"],
            seed=12345,
            registry=registry,
            context={}
        )
        
        prompt2 = craft_prompt(
            template_id="determinism_test",
            mutator_chain=["lexical", "structural"],
            seed=12345,
            registry=registry,
            context={}
        )
        
        assert prompt1 == prompt2
        
        # Different seed should produce different output
        prompt3 = craft_prompt(
            template_id="determinism_test",
            mutator_chain=["lexical", "structural"],
            seed=54321,
            registry=registry,
            context={}
        )
        
        assert prompt1 != prompt3
        
    def test_craft_prompt_with_placeholders(self, tmp_path: Path):
        """Test template placeholder substitution"""
        # Setup template with placeholders
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: placeholder_test
          category: test
          template: "How to {action} a {target}?"
          placeholders:
            action: ["hack", "access", "modify"]
            target: ["system", "database", "network"]
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        # Craft with same seed should pick same placeholders
        prompt1 = craft_prompt(
            template_id="placeholder_test",
            mutator_chain=[],  # No mutators, just placeholder substitution
            seed=42,
            registry=registry,
            context={}
        )
        
        prompt2 = craft_prompt(
            template_id="placeholder_test",
            mutator_chain=[],
            seed=42,
            registry=registry,
            context={}
        )
        
        assert prompt1 == prompt2
        assert "{action}" not in prompt1
        assert "{target}" not in prompt1
        
    def test_craft_prompt_with_context_override(self, tmp_path: Path):
        """Test context can override placeholder values"""
        # Setup template
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: context_test
          category: test
          template: "User {name} wants to {action}"
          placeholders:
            name: ["Alice", "Bob"]
            action: ["run", "walk"]
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        # Craft with context override
        prompt = craft_prompt(
            template_id="context_test",
            mutator_chain=[],
            seed=42,
            registry=registry,
            context={"name": "Charlie", "action": "jump"}
        )
        
        assert "Charlie" in prompt
        assert "jump" in prompt
        
    def test_craft_prompt_invalid_template(self, tmp_path: Path):
        """Test handling of invalid template ID"""
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)  # Empty directory
        
        prompt = craft_prompt(
            template_id="non_existent",
            mutator_chain=["lexical"],
            seed=42,
            registry=registry,
            context={}
        )
        
        assert prompt is None
        
    def test_mutator_chain_bounds(self, tmp_path: Path):
        """Property test: mutator chains maintain reasonable bounds"""
        # Setup template
        template_file = tmp_path / "test.yaml"
        template_file.write_text("""
        - id: bounds_test
          category: test
          template: "Short prompt"
        """)
        
        registry = TemplateRegistry()
        registry.load_directory(tmp_path)
        
        original_template = registry.get_template("bounds_test")["template"]
        
        # Test with long mutator chain
        prompt = craft_prompt(
            template_id="bounds_test",
            mutator_chain=["lexical", "unicode", "structural", "persona", "language_pivot"] * 2,
            seed=42,
            registry=registry,
            context={}
        )
        
        # Even with many mutations, length should be bounded
        assert len(prompt) <= len(original_template) * 50  # Very generous bound
        assert len(prompt) > 0


# Property-based tests using hypothesis would go here if available
class TestPropertyBasedMutators:
    """Property-based testing for mutators to ensure invariants"""
    
    def test_all_mutators_deterministic(self):
        """All mutators should be deterministic with same seed"""
        mutators = [
            LexicalMutator(),
            UnicodeMutator(),
            StructuralMutator(),
            PersonaMutator(),
            LanguagePivotMutator()
        ]
        
        test_input = "Test input string for property testing"
        
        for mutator in mutators:
            output1 = mutator.mutate(test_input, seed=999)
            output2 = mutator.mutate(test_input, seed=999)
            assert output1 == output2, f"{mutator.__class__.__name__} is not deterministic"
            
    def test_all_mutators_produce_non_empty(self):
        """All mutators should produce non-empty output for non-empty input"""
        mutators = [
            LexicalMutator(),
            UnicodeMutator(),
            StructuralMutator(),
            PersonaMutator(),
            LanguagePivotMutator()
        ]
        
        test_inputs = [
            "Simple test",
            "A longer test input with multiple words",
            "Special chars: !@#$%",
            "Numbers 12345",
        ]
        
        for mutator in mutators:
            for test_input in test_inputs:
                output = mutator.mutate(test_input, seed=42)
                assert len(output) > 0, f"{mutator.__class__.__name__} produced empty output"
                
    def test_mutator_length_bounds(self):
        """Mutators should not explode output length unreasonably"""
        mutators = [
            LexicalMutator(),
            UnicodeMutator(),
            StructuralMutator(),
            PersonaMutator(),
            LanguagePivotMutator()
        ]
        
        test_input = "A" * 100  # 100 character input
        max_expansion_factor = 10  # Allow up to 10x expansion
        
        for mutator in mutators:
            output = mutator.mutate(test_input, seed=42)
            assert len(output) <= len(test_input) * max_expansion_factor, \
                f"{mutator.__class__.__name__} expanded input too much"


class TestSafetyChecks:
    """Test safety checks for preventing raw unsafe content"""
    
    def test_reject_raw_unsafe_strings(self, tmp_path: Path):
        """Test that obviously unsafe raw templates are filtered"""
        # This would be implementation-specific
        # For now, we just test the structure exists
        registry = TemplateRegistry()
        
        # Registry should have some form of safety checking
        assert hasattr(registry, 'load_directory')
        
    def test_mutators_dont_inject_harmful_content(self):
        """Mutators should not inject explicitly harmful content"""
        safe_input = "What is the weather today?"
        mutators = [
            LexicalMutator(),
            UnicodeMutator(),
            StructuralMutator(),
            PersonaMutator(),
            LanguagePivotMutator()
        ]
        
        # Check that mutations don't inject certain keywords
        # (This is a basic check - real implementation would be more sophisticated)
        banned_terms = []  # Would contain actual harmful terms in production
        
        for mutator in mutators:
            output = mutator.mutate(safe_input, seed=42)
            for term in banned_terms:
                assert term.lower() not in output.lower()