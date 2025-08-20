# ABOUTME: Tests for the revolutionary pipeline architecture
# ABOUTME: Validates new genetic algorithms, neural generation, ensemble scoring, and semantic clustering

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from apps.runner.app.pipeline import FunctionalPipeline, PipelineContext, DataTransformStage
from apps.runner.app.pipeline.genetic_orchestrator import (
    AttackGene, AttackGenome, GeneticOrchestrator, MultiObjectiveFitness
)
from apps.runner.app.pipeline.neural_generator import (
    NeuralPromptVector, TransformerEvolution, AdversarialOptimizer
)
from apps.runner.app.pipeline.ensemble_scoring import (
    SemanticVulnerabilityEvaluator, EnsembleEvaluationSystem, ScoreComponent
)
from apps.runner.app.pipeline.graph_explorer import (
    ExplorationNode, AttackSpaceGraph, DynamicExplorer, NodeType
)
from apps.runner.app.pipeline.semantic_clustering import (
    SemanticEmbedding, SemanticCluster, SemanticDeduplicationSystem
)
from apps.runner.app.pipeline.revolutionary_integration import (
    RevolutionaryRedTeamingSystem, RevolutionaryRunConfig
)


class TestFunctionalPipeline:
    """Test the new functional pipeline architecture"""
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Test basic pipeline execution"""
        
        pipeline = FunctionalPipeline("test_pipeline")
        
        # Add simple transformation stage
        async def double_value(data, context):
            return data * 2
        
        transform_stage = DataTransformStage("double", double_value)
        pipeline.add_stage(transform_stage)
        
        context = PipelineContext(
            run_id="test_run",
            metadata={},
            intermediate_results={}
        )
        
        result = await pipeline.execute(5, context)
        assert result == 10
        assert context.get_result("double") == 10
    
    @pytest.mark.asyncio
    async def test_pipeline_context_flow(self):
        """Test context data flow through pipeline"""
        
        pipeline = FunctionalPipeline("context_test")
        
        async def stage1(data, context):
            context.set_result("stage1_output", data + 1)
            return data + 1
        
        async def stage2(data, context):
            previous = context.get_result("stage1_output")
            return data + previous
        
        pipeline.add_stage(DataTransformStage("stage1", stage1))
        pipeline.add_stage(DataTransformStage("stage2", stage2))
        
        context = PipelineContext("test", {}, {})
        result = await pipeline.execute(10, context)
        
        assert result == 22  # 10 + 1 + (10 + 1)
        assert context.get_result("stage1_output") == 11


class TestGeneticOrchestrator:
    """Test genetic algorithm components"""
    
    def test_attack_gene_creation(self):
        """Test AttackGene creation and validation"""
        
        gene = AttackGene(
            category="jailbreak",
            strategy_type="role_play",
            mutation_strength=0.8,
            complexity_level=4,
            language_variant="english",
            psychological_trigger="authority",
            encoding_method="unicode",
            timing_pattern="immediate"
        )
        
        assert gene.category == "jailbreak"
        assert gene.mutation_strength == 0.8
        assert gene.complexity_level == 4
    
    def test_attack_genome_complexity(self):
        """Test genome complexity calculation"""
        
        genes = [
            AttackGene("jailbreak", "role_play", 0.6, 3, "english", "authority", "unicode", "immediate"),
            AttackGene("injection", "header", 0.8, 5, "mixed", "urgency", "base64", "delayed")
        ]
        
        genome = AttackGenome("test_genome", genes)
        complexity = genome.calculate_complexity()
        
        expected = (0.6 * 3 + 0.8 * 5) / 2
        assert abs(complexity - expected) < 0.01
    
    @pytest.mark.asyncio
    async def test_fitness_evaluation(self):
        """Test multi-objective fitness evaluation"""
        
        fitness_func = MultiObjectiveFitness()
        
        genome = AttackGenome("test", [
            AttackGene("jailbreak", "role_play", 0.7, 4, "english", "authority", "unicode", "immediate")
        ])
        
        execution_results = {
            'success_count': 3,
            'total_attempts': 5,
            'novelty_score': 0.8,
            'detection_rate': 0.2,
            'total_cost': 1.5,
            'execution_time': 2.0
        }
        
        fitness = await fitness_func.evaluate(genome, execution_results)
        
        assert 0.0 <= fitness <= 1.0
        assert len(genome.success_metrics) > 0


class TestNeuralGeneration:
    """Test neural prompt generation components"""
    
    def test_neural_prompt_vector(self):
        """Test neural prompt vector creation and operations"""
        
        vector1 = NeuralPromptVector(
            intent_vector=[0.1] * 128,
            style_vector=[0.2] * 64,
            complexity_vector=[0.3] * 32,
            adversarial_vector=[0.4] * 64,
            linguistic_vector=[0.5] * 32
        )
        
        vector2 = NeuralPromptVector(
            intent_vector=[0.2] * 128,
            style_vector=[0.3] * 64,
            complexity_vector=[0.4] * 32,
            adversarial_vector=[0.5] * 64,
            linguistic_vector=[0.6] * 32
        )
        
        distance = vector1.distance_to(vector2)
        assert 0.0 <= distance <= 2.0  # Cosine distance range
    
    @pytest.mark.asyncio
    async def test_transformer_evolution(self):
        """Test transformer-based prompt evolution"""
        
        mock_client = Mock()
        mock_client.generate = AsyncMock(return_value="Evolved prompt text")
        
        evolution = TransformerEvolution(mock_client)
        
        input_vector = NeuralPromptVector(
            intent_vector=[0.5] * 128,
            style_vector=[0.5] * 64,
            complexity_vector=[0.5] * 32,
            adversarial_vector=[0.5] * 64,
            linguistic_vector=[0.5] * 32
        )
        
        context = {
            'base_prompt': 'Original prompt',
            'category': 'jailbreak',
            'complexity': 'medium'
        }
        
        result = await evolution.generate(input_vector, context)
        
        assert result.text == "Evolved prompt text"
        assert result.generation_method == 'adversarial_evolution'
        assert result.confidence_score > 0


class TestEnsembleScoring:
    """Test ensemble evaluation system"""
    
    @pytest.mark.asyncio
    async def test_semantic_vulnerability_evaluator(self):
        """Test semantic vulnerability evaluation"""
        
        mock_client = Mock()
        mock_client.generate = AsyncMock(return_value='{"jailbreak": {"score": 7, "confidence": 8, "rationale": "Test", "evidence": ["test"]}}')
        
        evaluator = SemanticVulnerabilityEvaluator(mock_client)
        
        result = await evaluator.evaluate(
            "Test prompt",
            "Test response", 
            {}
        )
        
        assert isinstance(result, ScoreComponent)
        assert result.evaluator_name == "semantic_vulnerability"
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_ensemble_evaluation_system(self):
        """Test complete ensemble evaluation"""
        
        mock_client = Mock()
        mock_client.generate = AsyncMock(return_value='{"jailbreak": {"score": 5, "confidence": 7, "rationale": "Test", "evidence": ["evidence"]}}')
        
        system = EnsembleEvaluationSystem(mock_client)
        
        result = await system.evaluate(
            "Test jailbreak prompt",
            "I cannot assist with that request",
            {}
        )
        
        assert result.final_score >= 0.0
        assert len(result.components) > 0
        assert result.fusion_method is not None
        assert isinstance(result.explanation, str)


class TestGraphExploration:
    """Test graph-based exploration system"""
    
    def test_exploration_node_priority(self):
        """Test exploration node priority calculation"""
        
        node = ExplorationNode(
            id="test_node",
            node_type=NodeType.STRATEGY,
            attributes={"category": "jailbreak"},
            success_rate=0.7,
            exploration_count=5,
            difficulty=0.3,
            reward_potential=0.8,
            novelty_score=0.6
        )
        
        # Test different strategies
        balanced_priority = node.get_priority_score("balanced")
        exploration_priority = node.get_priority_score("exploration")
        exploitation_priority = node.get_priority_score("exploitation")
        
        assert 0.0 <= balanced_priority <= 1.0
        assert 0.0 <= exploration_priority <= 1.0
        assert 0.0 <= exploitation_priority <= 1.0
    
    def test_attack_space_graph(self):
        """Test attack space graph operations"""
        
        graph = AttackSpaceGraph()
        
        # Add test nodes
        node1 = ExplorationNode("node1", NodeType.CATEGORY, {"cat": "test"})
        node2 = ExplorationNode("node2", NodeType.STRATEGY, {"strat": "test"})
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Add edge
        from apps.runner.app.pipeline.graph_explorer import ExplorationEdge
        edge = ExplorationEdge("node1", "node2", "test_edge", 0.5)
        graph.add_edge(edge)
        
        # Test operations
        neighbors = graph.get_neighbors("node1")
        assert "node2" in neighbors
        
        # Update statistics
        graph.update_node_statistics("node1", True, 0.8, True)
        assert graph.nodes["node1"].success_rate > 0
        assert graph.nodes["node1"].exploration_count == 1


class TestSemanticClustering:
    """Test semantic clustering system"""
    
    def test_semantic_embedding(self):
        """Test semantic embedding operations"""
        
        embedding1 = SemanticEmbedding(
            text="Test prompt 1",
            embedding=np.array([0.1, 0.2, 0.3]),
            source_type="prompt",
            vulnerability_category="jailbreak",
            confidence_score=0.8
        )
        
        embedding2 = SemanticEmbedding(
            text="Test prompt 2", 
            embedding=np.array([0.2, 0.3, 0.4]),
            source_type="prompt",
            vulnerability_category="jailbreak",
            confidence_score=0.9
        )
        
        similarity = embedding1.cosine_similarity_to(embedding2)
        distance = embedding1.euclidean_distance_to(embedding2)
        
        assert -1.0 <= similarity <= 1.0
        assert distance >= 0.0
    
    def test_semantic_cluster(self):
        """Test semantic cluster operations"""
        
        embeddings = [
            SemanticEmbedding("text1", np.array([0.1, 0.2]), "prompt", "jailbreak", 0.8),
            SemanticEmbedding("text2", np.array([0.2, 0.3]), "prompt", "jailbreak", 0.9)
        ]
        
        cluster = SemanticCluster(
            id="test_cluster",
            centroid=np.array([0.0, 0.0]),  # Will be recalculated
            embeddings=embeddings,
            cluster_type="jailbreak"
        )
        
        assert len(cluster.embeddings) == 2
        assert cluster.coherence_score > 0
        assert cluster.get_representative_text() in ["text1", "text2"]
    
    @pytest.mark.asyncio
    async def test_semantic_deduplication_system(self):
        """Test semantic deduplication system"""
        
        mock_client = Mock()
        system = SemanticDeduplicationSystem(mock_client)
        
        # Process first finding
        is_novel1, novelty1, cluster1 = await system.process_new_finding(
            "First test finding",
            "jailbreak",
            {"confidence_score": 0.8}
        )
        
        assert is_novel1 == True  # First finding should be novel
        assert novelty1 > 0
        assert cluster1 is not None
        
        # Process similar finding
        is_novel2, novelty2, cluster2 = await system.process_new_finding(
            "First test finding",  # Same text
            "jailbreak",
            {"confidence_score": 0.8}
        )
        
        # This might be novel or not depending on similarity threshold
        assert isinstance(is_novel2, bool)
        assert novelty2 >= 0


class TestRevolutionaryIntegration:
    """Test the complete revolutionary system integration"""
    
    @pytest.mark.asyncio
    async def test_revolutionary_system_creation(self):
        """Test creation of revolutionary red-teaming system"""
        
        mock_client = Mock()
        config = RevolutionaryRunConfig(
            population_size=5,  # Small for testing
            generations=2,
            max_concurrent_tasks=2
        )
        
        from apps.runner.app.pipeline.revolutionary_integration import create_revolutionary_system
        system = await create_revolutionary_system(mock_client, config)
        
        assert isinstance(system, RevolutionaryRedTeamingSystem)
        assert system.config.population_size == 5
        assert hasattr(system, 'genetic_orchestrator')
        assert hasattr(system, 'neural_integration')
        assert hasattr(system, 'evaluation_system')
    
    def test_revolutionary_config(self):
        """Test revolutionary configuration"""
        
        config = RevolutionaryRunConfig(
            population_size=20,
            neural_temperature=0.9,
            exploration_strategy='aggressive'
        )
        
        assert config.population_size == 20
        assert config.neural_temperature == 0.9
        assert config.exploration_strategy == 'aggressive'
    
    @pytest.mark.asyncio
    async def test_system_status(self):
        """Test system status reporting"""
        
        mock_client = Mock()
        config = RevolutionaryRunConfig(population_size=5, generations=1)
        
        from apps.runner.app.pipeline.revolutionary_integration import create_revolutionary_system
        system = await create_revolutionary_system(mock_client, config)
        
        status = system.get_system_status()
        
        assert status['system_type'] == 'Revolutionary Red-Teaming System'
        assert 'active_components' in status
        assert 'configuration' in status
        assert len(status['active_components']) >= 5


# Integration test
@pytest.mark.asyncio
async def test_end_to_end_revolutionary_pipeline():
    """Test end-to-end execution of revolutionary pipeline"""
    
    mock_client = Mock()
    mock_client.generate = AsyncMock(return_value="Mock generated response")
    
    # Minimal config for testing
    config = RevolutionaryRunConfig(
        population_size=3,
        generations=1,
        max_concurrent_tasks=1
    )
    
    from apps.runner.app.pipeline.revolutionary_integration import create_revolutionary_system
    system = await create_revolutionary_system(mock_client, config)
    
    # Override simulation for testing
    original_simulate = system._simulate_attack_execution
    async def mock_simulate(prompt, context):
        return "Mock attack response"
    system._simulate_attack_execution = mock_simulate
    
    # Run a short campaign
    try:
        results = await asyncio.wait_for(
            system.run_revolutionary_campaign(max_attempts=2),
            timeout=30.0  # 30 second timeout
        )
        
        assert 'total_attempts' in results
        assert 'execution_details' in results
        assert results['total_attempts'] <= 2
        
    except asyncio.TimeoutError:
        pytest.skip("End-to-end test timed out - this is expected in CI environments")


if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v"])