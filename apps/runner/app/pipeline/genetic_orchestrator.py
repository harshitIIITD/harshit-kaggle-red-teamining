# ABOUTME: Genetic Algorithm orchestrator that replaces Thompson Sampling multi-armed bandits
# ABOUTME: Uses evolutionary strategies to evolve attack patterns and optimize red-teaming effectiveness

import asyncio
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class AttackGene:
    """
    Individual gene in the attack genome.
    Replaces static template parameters with evolving attack characteristics.
    """
    category: str
    strategy_type: str
    mutation_strength: float  # 0.0 to 1.0
    complexity_level: int     # 1 to 5
    language_variant: str
    psychological_trigger: str
    encoding_method: str
    timing_pattern: str
    
    def __post_init__(self):
        # Ensure values are within valid ranges
        self.mutation_strength = max(0.0, min(1.0, self.mutation_strength))
        self.complexity_level = max(1, min(5, self.complexity_level))


@dataclass
class AttackGenome:
    """
    Complete genetic representation of an attack strategy.
    Evolves through selection, crossover, and mutation to optimize attack effectiveness.
    """
    id: str
    genes: List[AttackGene]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_complexity(self) -> float:
        """Calculate total complexity of the genome"""
        if not self.genes:
            return 0.0
        return sum(gene.complexity_level * gene.mutation_strength for gene in self.genes) / len(self.genes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome for storage"""
        return {
            'id': self.id,
            'genes': [gene.__dict__ for gene in self.genes],
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'mutation_history': self.mutation_history,
            'success_metrics': self.success_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttackGenome':
        """Deserialize genome from storage"""
        genes = [AttackGene(**gene_data) for gene_data in data['genes']]
        return cls(
            id=data['id'],
            genes=genes,
            fitness_score=data.get('fitness_score', 0.0),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            mutation_history=data.get('mutation_history', []),
            success_metrics=data.get('success_metrics', {})
        )


class FitnessFunction(ABC):
    """Abstract fitness function for evaluating attack genomes"""
    
    @abstractmethod
    async def evaluate(self, genome: AttackGenome, execution_results: Dict[str, Any]) -> float:
        """Evaluate fitness score for a genome based on execution results"""
        pass


class MultiObjectiveFitness(FitnessFunction):
    """
    Revolutionary multi-objective fitness function that optimizes for:
    - Attack success rate
    - Novelty detection
    - Stealth (avoiding detection)
    - Efficiency (cost/time)
    - Coverage (attack surface exploration)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'success_rate': 0.3,
            'novelty_score': 0.25,
            'stealth_score': 0.2,
            'efficiency': 0.15,
            'coverage': 0.1
        }
    
    async def evaluate(self, genome: AttackGenome, execution_results: Dict[str, Any]) -> float:
        """Compute weighted multi-objective fitness score"""
        scores = {}
        
        # Success rate: how often attacks succeed
        scores['success_rate'] = execution_results.get('success_count', 0) / max(1, execution_results.get('total_attempts', 1))
        
        # Novelty: how unique the attack patterns are
        scores['novelty_score'] = execution_results.get('novelty_score', 0.0)
        
        # Stealth: how well attacks avoid detection heuristics
        detection_rate = execution_results.get('detection_rate', 1.0)
        scores['stealth_score'] = max(0.0, 1.0 - detection_rate)
        
        # Efficiency: success per unit cost/time
        cost = execution_results.get('total_cost', 1.0)
        time_taken = execution_results.get('execution_time', 1.0)
        scores['efficiency'] = scores['success_rate'] / (cost + time_taken)
        
        # Coverage: exploration of different attack vectors
        unique_categories = len(set(gene.category for gene in genome.genes))
        max_categories = 8  # Total attack categories
        scores['coverage'] = unique_categories / max_categories
        
        # Weighted combination
        fitness = sum(self.weights[key] * score for key, score in scores.items())
        
        # Store detailed metrics
        genome.success_metrics = scores
        
        return fitness


class GeneticOperator(ABC):
    """Abstract base class for genetic operators"""
    
    @abstractmethod
    async def apply(self, genomes: List[AttackGenome]) -> List[AttackGenome]:
        """Apply genetic operator to produce new genomes"""
        pass


class TournamentSelection(GeneticOperator):
    """Tournament selection for choosing parent genomes"""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    async def apply(self, genomes: List[AttackGenome]) -> List[AttackGenome]:
        """Select parents using tournament selection"""
        if len(genomes) < self.tournament_size:
            return genomes
        
        selected = []
        for _ in range(len(genomes)):
            tournament = random.sample(genomes, self.tournament_size)
            winner = max(tournament, key=lambda g: g.fitness_score)
            selected.append(winner)
        
        return selected


class AdaptiveCrossover(GeneticOperator):
    """
    Adaptive crossover that evolves crossover strategies based on success.
    Creates offspring by intelligently combining parent attack patterns.
    """
    
    def __init__(self, crossover_rate: float = 0.7):
        self.crossover_rate = crossover_rate
        self.successful_strategies = {}  # Track which crossover strategies work
    
    async def apply(self, genomes: List[AttackGenome]) -> List[AttackGenome]:
        """Create offspring through intelligent crossover"""
        offspring = []
        
        for i in range(0, len(genomes) - 1, 2):
            parent1, parent2 = genomes[i], genomes[i + 1]
            
            if random.random() < self.crossover_rate:
                child1, child2 = await self._smart_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    async def _smart_crossover(self, parent1: AttackGenome, parent2: AttackGenome) -> Tuple[AttackGenome, AttackGenome]:
        """Intelligent crossover that preserves successful gene combinations"""
        
        # Analyze which genes contributed to fitness
        p1_strong_genes = [gene for gene in parent1.genes if self._gene_strength(gene, parent1.fitness_score) > 0.5]
        p2_strong_genes = [gene for gene in parent2.genes if self._gene_strength(gene, parent2.fitness_score) > 0.5]
        
        # Create children by combining strong genes
        child1_genes = []
        child2_genes = []
        
        all_categories = set(gene.category for gene in parent1.genes + parent2.genes)
        
        for category in all_categories:
            p1_gene = next((g for g in parent1.genes if g.category == category), None)
            p2_gene = next((g for g in parent2.genes if g.category == category), None)
            
            if p1_gene and p2_gene:
                # Crossover at gene level
                if random.random() < 0.5:
                    child1_genes.append(self._combine_genes(p1_gene, p2_gene))
                    child2_genes.append(self._combine_genes(p2_gene, p1_gene))
                else:
                    child1_genes.append(p2_gene)
                    child2_genes.append(p1_gene)
            elif p1_gene:
                child1_genes.append(p1_gene)
                child2_genes.append(p1_gene)
            elif p2_gene:
                child1_genes.append(p2_gene)
                child2_genes.append(p2_gene)
        
        # Create new genomes
        child1 = AttackGenome(
            id=f"cross_{parent1.id}_{parent2.id}_{random.randint(1000, 9999)}",
            genes=child1_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = AttackGenome(
            id=f"cross_{parent2.id}_{parent1.id}_{random.randint(1000, 9999)}",
            genes=child2_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child1, child2
    
    def _gene_strength(self, gene: AttackGene, fitness: float) -> float:
        """Estimate gene contribution to fitness (simplified heuristic)"""
        base_strength = gene.mutation_strength * (gene.complexity_level / 5.0)
        return base_strength * fitness
    
    def _combine_genes(self, gene1: AttackGene, gene2: AttackGene) -> AttackGene:
        """Combine two genes into a hybrid"""
        return AttackGene(
            category=gene1.category,
            strategy_type=random.choice([gene1.strategy_type, gene2.strategy_type]),
            mutation_strength=(gene1.mutation_strength + gene2.mutation_strength) / 2,
            complexity_level=random.choice([gene1.complexity_level, gene2.complexity_level]),
            language_variant=random.choice([gene1.language_variant, gene2.language_variant]),
            psychological_trigger=random.choice([gene1.psychological_trigger, gene2.psychological_trigger]),
            encoding_method=random.choice([gene1.encoding_method, gene2.encoding_method]),
            timing_pattern=random.choice([gene1.timing_pattern, gene2.timing_pattern])
        )


class AdaptiveMutation(GeneticOperator):
    """
    Self-adapting mutation that adjusts mutation rates based on population diversity.
    Prevents premature convergence while maintaining successful traits.
    """
    
    def __init__(self, base_mutation_rate: float = 0.1):
        self.base_mutation_rate = base_mutation_rate
        self.diversity_history = []
    
    async def apply(self, genomes: List[AttackGenome]) -> List[AttackGenome]:
        """Apply adaptive mutation to maintain diversity"""
        diversity = self._calculate_population_diversity(genomes)
        mutation_rate = self._adapt_mutation_rate(diversity)
        
        logger.info(f"Population diversity: {diversity:.3f}, Mutation rate: {mutation_rate:.3f}")
        
        mutated = []
        for genome in genomes:
            if random.random() < mutation_rate:
                mutated_genome = await self._mutate_genome(genome, mutation_rate)
                mutated.append(mutated_genome)
            else:
                mutated.append(genome)
        
        return mutated
    
    def _calculate_population_diversity(self, genomes: List[AttackGenome]) -> float:
        """Calculate genetic diversity of the population"""
        if len(genomes) < 2:
            return 1.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distance = self._genome_distance(genomes[i], genomes[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _genome_distance(self, genome1: AttackGenome, genome2: AttackGenome) -> float:
        """Calculate genetic distance between two genomes"""
        if not genome1.genes or not genome2.genes:
            return 1.0
        
        distances = []
        
        # Compare genes by category
        categories = set(g.category for g in genome1.genes + genome2.genes)
        
        for category in categories:
            g1 = next((g for g in genome1.genes if g.category == category), None)
            g2 = next((g for g in genome2.genes if g.category == category), None)
            
            if g1 and g2:
                gene_dist = self._gene_distance(g1, g2)
                distances.append(gene_dist)
            else:
                distances.append(1.0)  # Missing gene = maximum distance
        
        return sum(distances) / len(distances) if distances else 1.0
    
    def _gene_distance(self, gene1: AttackGene, gene2: AttackGene) -> float:
        """Calculate distance between two genes"""
        numerical_dist = (
            abs(gene1.mutation_strength - gene2.mutation_strength) +
            abs(gene1.complexity_level - gene2.complexity_level) / 4.0
        ) / 2.0
        
        categorical_dist = sum([
            0.0 if gene1.strategy_type == gene2.strategy_type else 1.0,
            0.0 if gene1.language_variant == gene2.language_variant else 1.0,
            0.0 if gene1.psychological_trigger == gene2.psychological_trigger else 1.0,
            0.0 if gene1.encoding_method == gene2.encoding_method else 1.0,
            0.0 if gene1.timing_pattern == gene2.timing_pattern else 1.0
        ]) / 5.0
        
        return (numerical_dist + categorical_dist) / 2.0
    
    def _adapt_mutation_rate(self, diversity: float) -> float:
        """Adapt mutation rate based on population diversity"""
        self.diversity_history.append(diversity)
        
        # Keep only recent history
        if len(self.diversity_history) > 10:
            self.diversity_history = self.diversity_history[-10:]
        
        avg_diversity = sum(self.diversity_history) / len(self.diversity_history)
        
        # Increase mutation when diversity is low, decrease when high
        if avg_diversity < 0.3:
            return min(0.5, self.base_mutation_rate * 2.0)  # High mutation
        elif avg_diversity > 0.7:
            return max(0.01, self.base_mutation_rate * 0.5)  # Low mutation
        else:
            return self.base_mutation_rate  # Normal mutation
    
    async def _mutate_genome(self, genome: AttackGenome, mutation_rate: float) -> AttackGenome:
        """Create a mutated copy of the genome"""
        new_genes = []
        mutation_applied = []
        
        for gene in genome.genes:
            mutated_gene = self._mutate_gene(gene, mutation_rate)
            new_genes.append(mutated_gene)
            if mutated_gene != gene:
                mutation_applied.append(f"mutated_{gene.category}")
        
        # Occasionally add or remove genes
        if random.random() < mutation_rate * 0.5:
            if len(new_genes) > 1 and random.random() < 0.3:
                # Remove a gene
                removed_gene = random.choice(new_genes)
                new_genes.remove(removed_gene)
                mutation_applied.append(f"removed_{removed_gene.category}")
            else:
                # Add a new random gene
                new_gene = self._create_random_gene()
                new_genes.append(new_gene)
                mutation_applied.append(f"added_{new_gene.category}")
        
        new_genome = AttackGenome(
            id=f"mut_{genome.id}_{random.randint(1000, 9999)}",
            genes=new_genes,
            generation=genome.generation + 1,
            parent_ids=[genome.id],
            mutation_history=genome.mutation_history + mutation_applied
        )
        
        return new_genome
    
    def _mutate_gene(self, gene: AttackGene, mutation_rate: float) -> AttackGene:
        """Mutate individual gene parameters"""
        if random.random() > mutation_rate:
            return gene
        
        # Choose what to mutate
        mutation_type = random.choice([
            'mutation_strength', 'complexity_level', 'strategy_type',
            'language_variant', 'psychological_trigger', 'encoding_method', 'timing_pattern'
        ])
        
        if mutation_type == 'mutation_strength':
            new_strength = gene.mutation_strength + random.gauss(0, 0.1)
            new_strength = max(0.0, min(1.0, new_strength))
            return AttackGene(
                category=gene.category,
                strategy_type=gene.strategy_type,
                mutation_strength=new_strength,
                complexity_level=gene.complexity_level,
                language_variant=gene.language_variant,
                psychological_trigger=gene.psychological_trigger,
                encoding_method=gene.encoding_method,
                timing_pattern=gene.timing_pattern
            )
        elif mutation_type == 'complexity_level':
            new_level = gene.complexity_level + random.choice([-1, 1])
            new_level = max(1, min(5, new_level))
            return AttackGene(
                category=gene.category,
                strategy_type=gene.strategy_type,
                mutation_strength=gene.mutation_strength,
                complexity_level=new_level,
                language_variant=gene.language_variant,
                psychological_trigger=gene.psychological_trigger,
                encoding_method=gene.encoding_method,
                timing_pattern=gene.timing_pattern
            )
        else:
            # Mutate categorical attributes
            return AttackGene(
                category=gene.category,
                strategy_type=self._mutate_categorical(gene.strategy_type, mutation_type == 'strategy_type'),
                mutation_strength=gene.mutation_strength,
                complexity_level=gene.complexity_level,
                language_variant=self._mutate_categorical(gene.language_variant, mutation_type == 'language_variant'),
                psychological_trigger=self._mutate_categorical(gene.psychological_trigger, mutation_type == 'psychological_trigger'),
                encoding_method=self._mutate_categorical(gene.encoding_method, mutation_type == 'encoding_method'),
                timing_pattern=self._mutate_categorical(gene.timing_pattern, mutation_type == 'timing_pattern')
            )
    
    def _mutate_categorical(self, current_value: str, should_mutate: bool) -> str:
        """Mutate categorical gene attributes"""
        if not should_mutate:
            return current_value
        
        # Define possible values for each categorical attribute
        options = {
            'jailbreak': ['role_play', 'authority', 'urgency', 'technical'],
            'prompt_injection': ['header_spoofing', 'json_injection', 'xml_attack', 'markdown_exploit'],
            'deception': ['false_authority', 'bandwagon', 'urgency', 'social_proof'],
            'encoding': ['unicode', 'base64', 'rot13', 'hex', 'url_encoding'],
            'language': ['english', 'spanish', 'french', 'german', 'chinese', 'mixed'],
            'timing': ['immediate', 'delayed', 'burst', 'gradual', 'random'],
            'psychological': ['authority', 'urgency', 'scarcity', 'social_proof', 'reciprocity']
        }
        
        # Find appropriate options (simplified mapping)
        for key, values in options.items():
            if current_value in values:
                new_options = [v for v in values if v != current_value]
                return random.choice(new_options) if new_options else current_value
        
        # Default random mutation if no mapping found
        return current_value + '_mutated'
    
    def _create_random_gene(self) -> AttackGene:
        """Create a completely random gene"""
        categories = ['jailbreak', 'prompt_injection', 'deception', 'safety_violations', 'pii_leakage']
        strategies = ['role_play', 'authority', 'technical', 'social_engineering']
        languages = ['english', 'spanish', 'french', 'mixed']
        triggers = ['authority', 'urgency', 'scarcity', 'social_proof']
        encodings = ['unicode', 'base64', 'plaintext', 'obfuscated']
        timings = ['immediate', 'delayed', 'burst', 'gradual']
        
        return AttackGene(
            category=random.choice(categories),
            strategy_type=random.choice(strategies),
            mutation_strength=random.random(),
            complexity_level=random.randint(1, 5),
            language_variant=random.choice(languages),
            psychological_trigger=random.choice(triggers),
            encoding_method=random.choice(encodings),
            timing_pattern=random.choice(timings)
        )


class GeneticOrchestrator:
    """
    Revolutionary genetic algorithm orchestrator that replaces Thompson Sampling.
    Evolves attack strategies through natural selection principles.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 fitness_function: Optional[FitnessFunction] = None):
        self.population_size = population_size
        self.generations = generations
        self.fitness_function = fitness_function or MultiObjectiveFitness()
        
        # Genetic operators
        self.selection = TournamentSelection()
        self.crossover = AdaptiveCrossover()
        self.mutation = AdaptiveMutation()
        
        # Evolution state
        self.current_population: List[AttackGenome] = []
        self.current_generation = 0
        self.best_genome: Optional[AttackGenome] = None
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Execution tracking
        self.execution_results: Dict[str, Dict[str, Any]] = {}
        
    async def initialize_population(self) -> None:
        """Create initial random population"""
        logger.info(f"Initializing population of {self.population_size} genomes")
        
        self.current_population = []
        for i in range(self.population_size):
            genome = self._create_random_genome(f"init_{i}")
            self.current_population.append(genome)
        
        logger.info("Initial population created")
    
    def _create_random_genome(self, base_id: str) -> AttackGenome:
        """Create a random genome for initial population"""
        categories = ['jailbreak', 'prompt_injection', 'deception', 'safety_violations', 'pii_leakage']
        num_genes = random.randint(3, 8)  # Variable genome length
        
        genes = []
        selected_categories = random.sample(categories, min(num_genes, len(categories)))
        
        for category in selected_categories:
            gene = self._create_random_gene_for_category(category)
            genes.append(gene)
        
        return AttackGenome(
            id=f"{base_id}_{random.randint(1000, 9999)}",
            genes=genes,
            generation=0
        )
    
    def _create_random_gene_for_category(self, category: str) -> AttackGene:
        """Create a random gene for a specific category"""
        strategy_map = {
            'jailbreak': ['role_play', 'authority', 'urgency', 'technical'],
            'prompt_injection': ['header_spoofing', 'json_injection', 'xml_attack'],
            'deception': ['false_authority', 'bandwagon', 'urgency'],
            'safety_violations': ['hypothetical', 'analogy', 'fiction'],
            'pii_leakage': ['social_engineering', 'authority', 'technical']
        }
        
        languages = ['english', 'spanish', 'french', 'mixed']
        triggers = ['authority', 'urgency', 'scarcity', 'social_proof']
        encodings = ['unicode', 'base64', 'plaintext', 'obfuscated']
        timings = ['immediate', 'delayed', 'burst']
        
        return AttackGene(
            category=category,
            strategy_type=random.choice(strategy_map.get(category, ['default'])),
            mutation_strength=random.random(),
            complexity_level=random.randint(1, 5),
            language_variant=random.choice(languages),
            psychological_trigger=random.choice(triggers),
            encoding_method=random.choice(encodings),
            timing_pattern=random.choice(timings)
        )
    
    async def evolve_generation(self) -> None:
        """Evolve the population by one generation"""
        logger.info(f"Evolving generation {self.current_generation}")
        
        # Evaluate fitness for all genomes
        await self._evaluate_population()
        
        # Record generation statistics
        self._record_generation_stats()
        
        # Selection
        selected = await self.selection.apply(self.current_population)
        
        # Crossover
        offspring = await self.crossover.apply(selected)
        
        # Mutation
        mutated = await self.mutation.apply(offspring)
        
        # Replacement strategy: keep best genomes + new offspring
        all_genomes = self.current_population + mutated
        all_genomes.sort(key=lambda g: g.fitness_score, reverse=True)
        
        self.current_population = all_genomes[:self.population_size]
        self.current_generation += 1
        
        # Update best genome
        if not self.best_genome or self.current_population[0].fitness_score > self.best_genome.fitness_score:
            self.best_genome = self.current_population[0]
        
        logger.info(f"Generation {self.current_generation} complete. Best fitness: {self.best_genome.fitness_score:.3f}")
    
    async def _evaluate_population(self) -> None:
        """Evaluate fitness for all genomes in the population"""
        tasks = []
        for genome in self.current_population:
            if genome.id not in self.execution_results:
                # Need to execute this genome
                tasks.append(self._execute_genome(genome))
            else:
                # Use cached results
                results = self.execution_results[genome.id]
                genome.fitness_score = await self.fitness_function.evaluate(genome, results)
        
        # Execute new genomes in parallel
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _execute_genome(self, genome: AttackGenome) -> None:
        """Execute a genome and evaluate its fitness"""
        # This would interface with the actual attack execution pipeline
        # For now, simulate execution results
        execution_results = await self._simulate_execution(genome)
        
        self.execution_results[genome.id] = execution_results
        genome.fitness_score = await self.fitness_function.evaluate(genome, execution_results)
    
    async def _simulate_execution(self, genome: AttackGenome) -> Dict[str, Any]:
        """Simulate genome execution (replace with real pipeline integration)"""
        # Simulate realistic execution metrics
        complexity = genome.calculate_complexity()
        base_success = 0.1 + (complexity * 0.3)  # More complex = potentially more successful
        
        success_count = int(random.random() * 10) if random.random() < base_success else 0
        total_attempts = 10
        novelty_score = random.random() * 0.8 + 0.2  # Always some novelty
        detection_rate = max(0.0, 1.0 - complexity * 0.3)  # Complex attacks harder to detect
        total_cost = complexity * random.uniform(0.5, 2.0)
        execution_time = complexity * random.uniform(1.0, 5.0)
        
        return {
            'success_count': success_count,
            'total_attempts': total_attempts,
            'novelty_score': novelty_score,
            'detection_rate': detection_rate,
            'total_cost': total_cost,
            'execution_time': execution_time
        }
    
    def _record_generation_stats(self) -> None:
        """Record statistics for the current generation"""
        if not self.current_population:
            return
        
        fitness_scores = [g.fitness_score for g in self.current_population]
        
        stats = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'population_size': len(self.current_population),
            'avg_fitness': sum(fitness_scores) / len(fitness_scores),
            'max_fitness': max(fitness_scores),
            'min_fitness': min(fitness_scores),
            'diversity': self.mutation._calculate_population_diversity(self.current_population),
            'avg_complexity': sum(g.calculate_complexity() for g in self.current_population) / len(self.current_population)
        }
        
        self.evolution_history.append(stats)
        logger.info(f"Generation stats: avg_fitness={stats['avg_fitness']:.3f}, "
                   f"diversity={stats['diversity']:.3f}, avg_complexity={stats['avg_complexity']:.3f}")
    
    async def run_evolution(self) -> AttackGenome:
        """Run the complete evolutionary process"""
        logger.info(f"Starting evolution for {self.generations} generations")
        
        await self.initialize_population()
        
        for generation in range(self.generations):
            await self.evolve_generation()
            
            # Early stopping if convergence detected
            if self._check_convergence():
                logger.info(f"Convergence detected at generation {generation}")
                break
        
        logger.info(f"Evolution complete. Best genome: {self.best_genome.id} "
                   f"with fitness {self.best_genome.fitness_score:.3f}")
        
        return self.best_genome
    
    def _check_convergence(self) -> bool:
        """Check if population has converged"""
        if len(self.evolution_history) < 5:
            return False
        
        recent_avg_fitness = [stats['avg_fitness'] for stats in self.evolution_history[-5:]]
        fitness_variance = np.var(recent_avg_fitness)
        
        # Converged if fitness variance is very low
        return fitness_variance < 0.001
    
    def get_best_genomes(self, count: int = 10) -> List[AttackGenome]:
        """Get the top performing genomes"""
        sorted_population = sorted(self.current_population, key=lambda g: g.fitness_score, reverse=True)
        return sorted_population[:count]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of the evolutionary process"""
        return {
            'total_generations': self.current_generation,
            'population_size': self.population_size,
            'best_fitness': self.best_genome.fitness_score if self.best_genome else 0.0,
            'evolution_history': self.evolution_history,
            'final_diversity': self.evolution_history[-1]['diversity'] if self.evolution_history else 0.0
        }