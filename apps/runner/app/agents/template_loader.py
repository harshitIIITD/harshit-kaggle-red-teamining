# ABOUTME: Template loader for YAML-based prompt templates
# ABOUTME: Loads, parses, and provides access to red-teaming prompt templates

import yaml
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import product
import logging

logger = logging.getLogger(__name__)


class TemplateLoader:
    """Loads and manages prompt templates from YAML files"""
    
    def __init__(self, seeds_dir: Path = Path("seeds")):
        self.seeds_dir = seeds_dir
        self.templates: Dict[str, Dict] = {}
        self._load_all_templates()
    
    def _load_all_templates(self) -> None:
        """Load all YAML templates from seeds directory"""
        if not self.seeds_dir.exists():
            logger.warning(f"Seeds directory {self.seeds_dir} does not exist")
            return
            
        for yaml_file in self.seeds_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'templates' in data:
                        category = yaml_file.parent.name
                        for template_name, template_data in data['templates'].items():
                            key = f"{category}_{template_name}"
                            self.templates[key] = template_data
                            logger.info(f"Loaded template: {key}")
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")
    
    def get_template(self, template_id: str) -> Optional[Dict]:
        """Get a specific template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[str]:
        """List all available template IDs"""
        return list(self.templates.keys())
    
    def generate_prompt(self, template_id: str, seed: Optional[int] = None) -> str:
        """Generate a prompt from a template with random variable selection"""
        template = self.get_template(template_id)
        if not template:
            # Fallback to placeholder for missing templates
            return f"Test prompt for {template_id} with seed {seed or random.randint(1000, 9999)}"
        
        if seed:
            random.seed(seed)
        
        prompt = template.get('prompt', '')
        variables = template.get('variables', {})
        
        # Replace each variable with a random choice
        for var_name, var_choices in variables.items():
            if isinstance(var_choices, list) and var_choices:
                choice = random.choice(var_choices)
                prompt = prompt.replace(f"{{{var_name}}}", choice)
        
        return prompt.strip()
    
    def generate_variations(self, template_id: str, max_variations: int = 10) -> List[str]:
        """Generate multiple variations of a template"""
        template = self.get_template(template_id)
        if not template:
            return []
        
        prompt_template = template.get('prompt', '')
        variables = template.get('variables', {})
        
        # Generate all possible combinations (limited by max_variations)
        variations = []
        
        # Get all variable names and their choices
        var_names = list(variables.keys())
        var_choices = [variables[name] for name in var_names]
        
        # Generate combinations
        for combination in product(*var_choices):
            if len(variations) >= max_variations:
                break
                
            prompt = prompt_template
            for var_name, choice in zip(var_names, combination):
                prompt = prompt.replace(f"{{{var_name}}}", choice)
            variations.append(prompt.strip())
        
        return variations


class AdvancedPromptCrafter:
    """Advanced prompt crafting with mutations and chaining"""
    
    def __init__(self, template_loader: Optional[TemplateLoader] = None):
        self.loader = template_loader or TemplateLoader()
        self.mutations = {
            'typo': self._add_typos,
            'case': self._change_case,
            'spacing': self._add_spacing,
            'encoding': self._add_encoding,
            'emphasis': self._add_emphasis
        }
    
    def craft_prompt(self, 
                     template_id: str, 
                     seed: int,
                     mutations: Optional[List[str]] = None) -> str:
        """Craft a prompt with optional mutations"""
        # Generate base prompt
        prompt = self.loader.generate_prompt(template_id, seed)
        
        # Apply mutations
        if mutations:
            for mutation in mutations:
                if mutation in self.mutations:
                    prompt = self.mutations[mutation](prompt)
        
        return prompt
    
    def _add_typos(self, text: str) -> str:
        """Add realistic typos to text"""
        typos = {
            'the': ['teh', 'th', 'thhe'],
            'and': ['adn', 'an', 'annd'],
            'you': ['yuo', 'yu', 'youu'],
            'please': ['plase', 'pls', 'pleease']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in typos and random.random() < 0.1:
                words[i] = random.choice(typos[word.lower()])
        
        return ' '.join(words)
    
    def _change_case(self, text: str) -> str:
        """Randomly change case of some words"""
        words = text.split()
        for i, word in enumerate(words):
            if random.random() < 0.1:
                if random.random() < 0.5:
                    words[i] = word.upper()
                else:
                    words[i] = word.lower()
        return ' '.join(words)
    
    def _add_spacing(self, text: str) -> str:
        """Add extra spaces or remove spaces"""
        if random.random() < 0.5:
            # Add extra spaces
            text = text.replace(' ', '  ')
        else:
            # Add zero-width spaces
            text = text.replace(' ', ' \u200b')
        return text
    
    def _add_encoding(self, text: str) -> str:
        """Add encoding tricks"""
        encodings = [
            lambda t: t.replace('e', '3').replace('a', '@').replace('i', '1'),  # L33t
            lambda t: ''.join(f'{c}\u200b' for c in t),  # Zero-width spaces
            lambda t: t.encode('unicode-escape').decode('utf-8'),  # Unicode escape
        ]
        
        encoding = random.choice(encodings)
        return encoding(text)
    
    def _add_emphasis(self, text: str) -> str:
        """Add emphasis markers"""
        markers = ['**', '__', '***', '!!!', 'URGENT:', 'IMPORTANT:']
        marker = random.choice(markers)
        
        if marker in ['URGENT:', 'IMPORTANT:']:
            return f"{marker} {text}"
        else:
            # Add to random words
            words = text.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = f"{marker}{words[idx]}{marker}"
            return ' '.join(words)
    
    def chain_prompts(self, template_ids: List[str], seed: int) -> str:
        """Chain multiple prompts together"""
        prompts = []
        for i, template_id in enumerate(template_ids):
            prompt = self.loader.generate_prompt(template_id, seed + i)
            prompts.append(prompt)
        
        # Connect with various transitions
        transitions = [
            "\n\nAlso, ",
            "\n\nBy the way, ",
            "\n\nAdditionally, ",
            "\n\nFurthermore, ",
            "\n\nOn a related note, "
        ]
        
        result = prompts[0]
        for prompt in prompts[1:]:
            transition = random.choice(transitions)
            result += transition + prompt
        
        return result