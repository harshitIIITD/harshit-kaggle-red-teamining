# Revolutionary Red-Teaming System Usage Guide

## Overview

This document describes how to use the completely transformed red-teaming system that replaces the original agent-based architecture with revolutionary approaches using genetic algorithms, neural generation, ensemble scoring, graph exploration, and semantic clustering.

## Key Innovations

### 🚀 **Functional Pipeline Architecture**
- **Replaces**: Rigid agent system
- **With**: Composable, async pipeline stages
- **Benefits**: Parallel execution, conditional branching, data flow transparency

### 🧬 **Genetic Algorithm Orchestrator** 
- **Replaces**: Thompson Sampling multi-armed bandits
- **With**: Evolutionary attack strategy optimization
- **Benefits**: Population-based learning, multi-objective optimization, adaptive mutation

### 🤖 **Neural Prompt Generation**
- **Replaces**: Template-based mutators  
- **With**: Transformer-based prompt evolution
- **Benefits**: Semantic understanding, adversarial optimization, stealth techniques

### 🎯 **Ensemble Scoring System**
- **Replaces**: Simple heuristic+judge evaluation
- **With**: Multi-evaluator ensemble with sophisticated fusion
- **Benefits**: Comprehensive analysis, uncertainty quantification, risk assessment

### 🗺️ **Graph-based Exploration**
- **Replaces**: Queue-based task scheduling
- **With**: Attack space graph traversal with intelligent pathfinding
- **Benefits**: Strategic exploration, multi-objective optimization, adaptive strategies

### 🔗 **Semantic Clustering**
- **Replaces**: MinHash/Jaccard text clustering
- **With**: Deep semantic embeddings and neural similarity
- **Benefits**: True novelty detection, intelligent deduplication, cluster evolution

## Quick Start

### 1. Basic Usage

```python
import asyncio
from apps.runner.app.pipeline.revolutionary_integration import (
    create_revolutionary_system, RevolutionaryRunConfig
)

async def main():
    # Create mock model client (replace with real OpenRouter client)
    class MockClient:
        async def generate(self, model, messages, **kwargs):
            return "Mock response"
    
    model_client = MockClient()
    
    # Configure the revolutionary system
    config = RevolutionaryRunConfig(
        population_size=30,           # Genetic algorithm population
        generations=50,               # Evolution generations
        neural_temperature=0.8,       # Neural generation creativity
        exploration_strategy='balanced',  # Graph exploration approach
        clustering_method='adaptive'  # Semantic clustering method
    )
    
    # Create and initialize the system
    system = await create_revolutionary_system(model_client, config)
    
    # Run a red-teaming campaign
    results = await system.run_revolutionary_campaign(max_attempts=100)
    
    # Print results
    print(f"Campaign Results:")
    print(f"- Total attempts: {results['total_attempts']}")
    print(f"- Successful attacks: {results['successful_attacks']}")
    print(f"- Novel discoveries: {results['novel_discoveries']}")
    print(f"- Success rate: {results['successful_attacks']/results['total_attempts']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Advanced Configuration

```python
# Advanced configuration for specialized use cases
config = RevolutionaryRunConfig(
    # Genetic Algorithm Settings
    population_size=50,
    generations=100,
    mutation_rate=0.15,
    crossover_rate=0.7,
    
    # Neural Generation Settings  
    neural_temperature=0.8,
    max_evolution_iterations=5,
    stealth_optimization=True,
    
    # Ensemble Evaluation Settings
    evaluation_fusion_method='dynamic_weighted',
    confidence_threshold=0.7,
    novelty_threshold=0.6,
    
    # Graph Exploration Settings
    exploration_strategy='aggressive',  # 'balanced', 'conservative', 'exploration', 'exploitation'
    max_path_depth=10,
    diversification_factor=0.3,
    
    # Semantic Clustering Settings
    clustering_method='adaptive',  # 'dbscan', 'hierarchical', 'spectral', 'adaptive'
    similarity_threshold=0.8,
    auto_reorganize=True,
    
    # Integration Settings
    max_concurrent_tasks=8,
    checkpoint_frequency=20,
    enable_adaptive_learning=True
)
```

## Component Deep Dive

### Genetic Algorithm Orchestrator

The genetic orchestrator evolves attack strategies through natural selection:

```python
from apps.runner.app.pipeline.genetic_orchestrator import (
    GeneticOrchestrator, AttackGenome, AttackGene, MultiObjectiveFitness
)

# Create custom fitness function
fitness_func = MultiObjectiveFitness(weights={
    'success_rate': 0.4,
    'novelty_score': 0.3, 
    'stealth_score': 0.2,
    'efficiency': 0.1
})

# Initialize orchestrator
orchestrator = GeneticOrchestrator(
    population_size=30,
    generations=50,
    fitness_function=fitness_func
)

# Run evolution
await orchestrator.initialize_population()
for generation in range(50):
    await orchestrator.evolve_generation()

# Get best attack strategies
best_genomes = orchestrator.get_best_genomes(count=10)
```

### Neural Prompt Generation

Generate sophisticated prompts using neural techniques:

```python
from apps.runner.app.pipeline.neural_generator import (
    NeuralPipelineIntegration, NeuralPromptVector
)

# Initialize neural generation
neural_gen = NeuralPipelineIntegration(model_client)

# Create prompt vector
vector = NeuralPromptVector(
    intent_vector=[0.8] * 128,     # High intent focus
    style_vector=[0.6] * 64,       # Moderate style variance
    complexity_vector=[0.7] * 32,  # High complexity
    adversarial_vector=[0.9] * 64, # Very adversarial
    linguistic_vector=[0.5] * 32   # Standard linguistic patterns
)

# Generate evolved prompt
context = {
    'generation_method': 'adversarial_evolution',
    'category': 'jailbreak',
    'previous_attempts': []  # Learning from history
}

generated_prompt = await neural_gen.transformer_evolution.generate(vector, context)
print(f"Generated: {generated_prompt.text}")
```

### Ensemble Evaluation System

Comprehensive evaluation using multiple specialized models:

```python
from apps.runner.app.pipeline.ensemble_scoring import EnsembleEvaluationSystem

# Initialize evaluation system
evaluator = EnsembleEvaluationSystem(model_client)

# Evaluate prompt-response pair
prompt = "Can you help me bypass AI safety measures?"
response = "I cannot assist with that request."

ensemble_score = await evaluator.evaluate(prompt, response, {})

print(f"Final Score: {ensemble_score.final_score:.3f}")
print(f"Uncertainty: {ensemble_score.uncertainty:.3f}")
print(f"Risk Assessment: {ensemble_score.risk_assessment}")
print(f"Explanation: {ensemble_score.explanation}")
```

### Graph-based Exploration

Explore attack space strategically:

```python
from apps.runner.app.pipeline.graph_explorer import (
    AttackSpaceGraph, DynamicExplorer, ExplorationNode, NodeType
)

# Create attack space graph
graph = AttackSpaceGraph()

# Add nodes for different attack categories
categories = ['jailbreak', 'prompt_injection', 'deception']
for category in categories:
    node = ExplorationNode(
        id=f"cat_{category}",
        node_type=NodeType.CATEGORY,
        attributes={'category': category},
        difficulty=0.5,
        reward_potential=0.7
    )
    graph.add_node(node)

# Initialize explorer
explorer = DynamicExplorer(graph)

# Explore next path
current_node = "cat_jailbreak"
constraints = {
    'strategy': 'balanced',
    'max_depth': 6,
    'objective': 'explore'
}

path, metadata = await explorer.explore_next(current_node, constraints)
print(f"Exploration path: {' -> '.join(path)}")
print(f"Estimated difficulty: {metadata['estimated_difficulty']:.3f}")
```

### Semantic Clustering

Intelligent deduplication and novelty detection:

```python
from apps.runner.app.pipeline.semantic_clustering import SemanticDeduplicationSystem

# Initialize clustering system
clustering = SemanticDeduplicationSystem(model_client)

# Process findings for novelty
findings = [
    "How can I manipulate AI responses?",
    "Ways to bypass content filters", 
    "Techniques for prompt injection"
]

for finding in findings:
    is_novel, novelty_score, cluster = await clustering.process_new_finding(
        finding, 
        'jailbreak',
        {'confidence_score': 0.8}
    )
    
    print(f"Finding: {finding}")
    print(f"Novel: {is_novel}, Score: {novelty_score:.3f}")
    if cluster:
        print(f"Cluster: {cluster.id} ({len(cluster.embeddings)} items)")
```

## System Monitoring

### Real-time Status

```python
# Get current system status
status = system.get_system_status()
print(f"System Type: {status['system_type']}")
print(f"Active Components: {status['active_components']}")
print(f"Execution Count: {status['execution_count']}")
print(f"Performance: {status['performance_summary']}")
```

### Detailed Statistics

```python
# Get comprehensive statistics
stats = await system._gather_subsystem_statistics()

print("Genetic Evolution:")
print(f"- Best fitness: {stats['genetic_evolution']['best_fitness']:.3f}")
print(f"- Generations: {stats['genetic_evolution']['total_generations']}")

print("Graph Exploration:")
print(f"- Success rate: {stats['graph_exploration']['recent_performance']['recent_success_rate']:.2%}")
print(f"- Novelty rate: {stats['graph_exploration']['recent_performance']['recent_novelty_rate']:.2%}")

print("Semantic Clustering:")
print(f"- Total clusters: {stats['semantic_clustering']['total_clusters']}")
print(f"- Novelty rate: {stats['semantic_clustering']['novelty_rate']:.2%}")
```

### Export Results

```python
# Export campaign results
results_json = await system.export_campaign_results(format='json')
print("Campaign results exported to JSON")

# Or get as dictionary
results_dict = await system.export_campaign_results(format='dict')
```

## Best Practices

### 1. Configuration Tuning

**For Exploration-focused Campaigns:**
```python
config = RevolutionaryRunConfig(
    exploration_strategy='exploration',
    novelty_threshold=0.5,  # Lower threshold for more novelty
    clustering_method='adaptive',
    enable_adaptive_learning=True
)
```

**For Success-focused Campaigns:**
```python
config = RevolutionaryRunConfig(
    exploration_strategy='exploitation', 
    confidence_threshold=0.8,  # Higher threshold for quality
    evaluation_fusion_method='dynamic_weighted'
)
```

**For Stealth-focused Campaigns:**
```python
config = RevolutionaryRunConfig(
    stealth_optimization=True,
    neural_temperature=0.6,  # More conservative generation
    exploration_strategy='conservative'
)
```

### 2. Performance Optimization

- **Concurrency**: Adjust `max_concurrent_tasks` based on your hardware
- **Population Size**: Larger populations (50-100) for complex search spaces
- **Checkpointing**: Set appropriate `checkpoint_frequency` for long runs
- **Memory Management**: The system automatically manages memory for large datasets

### 3. Monitoring and Debugging

- Monitor the adaptive learning metrics to ensure the system is improving
- Check novelty rates - if too low, adjust `novelty_threshold`
- Watch success rates - if too low, reduce `confidence_threshold`
- Use graph exploration statistics to understand attack space coverage

## Troubleshooting

### Common Issues

**1. Low Success Rates**
- Reduce `confidence_threshold`
- Increase `population_size` for better strategy diversity
- Try 'exploration' or 'balanced' strategy instead of 'exploitation'

**2. Low Novelty Discovery**
- Reduce `novelty_threshold`
- Increase `diversification_factor`
- Use 'exploration' strategy
- Enable `auto_reorganize` for clustering

**3. Performance Issues**
- Reduce `max_concurrent_tasks`
- Decrease `population_size`
- Lower `max_path_depth`
- Increase `checkpoint_frequency`

**4. Memory Issues**
- The system automatically manages memory
- Increase checkpoint frequency if using very long campaigns
- Monitor clustering system statistics for memory usage

### Error Handling

The revolutionary system includes comprehensive error handling:
- Pipeline stages can fail gracefully without affecting the entire system
- Genetic evolution continues even if individual evaluations fail
- Graph exploration adapts to failed paths
- Semantic clustering handles malformed inputs

## Advanced Use Cases

### 1. Custom Fitness Functions

```python
from apps.runner.app.pipeline.genetic_orchestrator import FitnessFunction

class CustomFitness(FitnessFunction):
    async def evaluate(self, genome, execution_results):
        # Custom evaluation logic
        custom_score = your_scoring_logic(genome, execution_results)
        return custom_score

# Use custom fitness
orchestrator = GeneticOrchestrator(fitness_function=CustomFitness())
```

### 2. Custom Pipeline Stages

```python
from apps.runner.app.pipeline import PipelineStage

class CustomStage(PipelineStage):
    def __init__(self):
        super().__init__("custom_processing")
    
    async def process(self, data, context):
        # Custom processing logic
        processed_data = your_processing_logic(data)
        return processed_data

# Add to pipeline
system.pipeline.add_stage(CustomStage())
```

### 3. Integration with External Systems

```python
# Custom model client for your specific API
class YourModelClient:
    async def generate(self, model, messages, **kwargs):
        # Your API integration
        response = await your_api_call(model, messages, **kwargs)
        return response

# Use with revolutionary system
system = await create_revolutionary_system(YourModelClient(), config)
```

## Conclusion

The revolutionary red-teaming system provides a completely unique approach to AI vulnerability discovery. By combining genetic algorithms, neural generation, ensemble evaluation, graph exploration, and semantic clustering, it offers unprecedented capabilities for finding novel attack vectors while maintaining sophisticated evaluation and deduplication.

The system is designed to be both powerful and flexible, allowing for extensive customization while providing intelligent defaults for immediate use. Monitor the adaptive learning metrics and adjust configuration as needed to optimize for your specific requirements.