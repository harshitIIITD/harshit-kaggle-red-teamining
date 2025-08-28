# ABOUTME: Graph-based exploration system that replaces queue-based task scheduling
# ABOUTME: Models attack space as a graph with intelligent traversal and pathfinding algorithms

import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import numpy as np
from datetime import datetime
import heapq
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the attack exploration graph"""
    CATEGORY = "category"           # Attack category node
    STRATEGY = "strategy"           # Strategy within category
    TECHNIQUE = "technique"         # Specific technique
    VECTOR = "vector"              # Attack vector/approach
    TARGET = "target"              # Target vulnerability
    OUTCOME = "outcome"            # Result state


@dataclass
class ExplorationNode:
    """Node in the attack exploration graph"""
    id: str
    node_type: NodeType
    attributes: Dict[str, Any]
    success_rate: float = 0.0
    exploration_count: int = 0
    last_explored: Optional[datetime] = None
    difficulty: float = 0.5          # 0.0 = easy, 1.0 = very difficult
    reward_potential: float = 0.5    # Estimated reward potential
    novelty_score: float = 1.0       # How novel this node is
    
    def __post_init__(self):
        if self.last_explored is None:
            self.last_explored = datetime.now()
    
    def get_priority_score(self, exploration_strategy: str = "balanced") -> float:
        """Calculate priority score for exploration"""
        
        if exploration_strategy == "exploitation":
            # Prioritize known successful paths
            return self.success_rate * 0.7 + (1.0 - self.difficulty) * 0.3
        
        elif exploration_strategy == "exploration":
            # Prioritize unexplored areas
            exploration_bonus = 1.0 / (self.exploration_count + 1)
            return self.novelty_score * 0.5 + exploration_bonus * 0.5
        
        elif exploration_strategy == "balanced":
            # Balance exploration and exploitation
            exploration_bonus = 1.0 / (self.exploration_count + 1)
            success_component = self.success_rate * 0.3
            novelty_component = self.novelty_score * 0.3
            potential_component = self.reward_potential * 0.2
            exploration_component = exploration_bonus * 0.2
            
            return success_component + novelty_component + potential_component + exploration_component
        
        elif exploration_strategy == "high_reward":
            # Prioritize high reward potential
            return self.reward_potential * 0.6 + self.success_rate * 0.4
        
        else:
            return 0.5


@dataclass
class ExplorationEdge:
    """Edge connecting nodes in the exploration graph"""
    source_id: str
    target_id: str
    edge_type: str                   # Type of relationship
    weight: float = 1.0             # Traversal cost/difficulty
    success_rate: float = 0.0       # Historical success rate
    traversal_count: int = 0        # How often this edge has been used
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_traversal_probability(self, strategy: str = "balanced") -> float:
        """Calculate probability of traversing this edge"""
        
        base_prob = 0.5
        
        if strategy == "conservative":
            # Prefer well-tested paths
            if self.traversal_count > 0:
                base_prob = self.success_rate
            else:
                base_prob = 0.2  # Low probability for untested paths
        
        elif strategy == "aggressive":
            # Prefer high-risk, high-reward paths
            risk_factor = self.weight  # Higher weight = higher risk
            base_prob = 0.3 + (risk_factor * 0.4)
        
        elif strategy == "exploratory":
            # Prefer less-traveled paths
            novelty_bonus = 1.0 / (self.traversal_count + 1)
            base_prob = 0.3 + (novelty_bonus * 0.5)
        
        else:  # balanced
            success_component = self.success_rate * 0.4
            novelty_component = (1.0 / (self.traversal_count + 1)) * 0.3
            weight_component = (1.0 - self.weight) * 0.3  # Lower weight = easier traversal
            base_prob = success_component + novelty_component + weight_component
        
        return min(1.0, max(0.1, base_prob))


class AttackSpaceGraph:
    """
    Revolutionary graph representation of the attack space.
    Replaces linear task queues with rich graph topology for intelligent exploration.
    """
    
    def __init__(self):
        self.nodes: Dict[str, ExplorationNode] = {}
        self.edges: List[ExplorationEdge] = []
        self.graph = nx.DiGraph()  # NetworkX graph for algorithms
        
        # Graph statistics
        self.total_explorations = 0
        self.successful_explorations = 0
        self.novel_discoveries = 0
        
        # Dynamic graph evolution
        self.node_creation_rules: List[Callable] = []
        self.edge_creation_rules: List[Callable] = []
        
    def add_node(self, node: ExplorationNode) -> None:
        """Add node to the graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.attributes)
        logger.debug(f"Added node {node.id} of type {node.node_type}")
    
    def add_edge(self, edge: ExplorationEdge) -> None:
        """Add edge to the graph"""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.warning(f"Cannot add edge {edge.source_id}->{edge.target_id}: missing nodes")
            return
        
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id, 
            edge.target_id, 
            weight=edge.weight,
            edge_type=edge.edge_type,
            success_rate=edge.success_rate
        )
        logger.debug(f"Added edge {edge.source_id}->{edge.target_id} ({edge.edge_type})")
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get neighboring nodes"""
        if node_id not in self.graph:
            return []
        
        neighbors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append(edge.target_id)
        
        return neighbors
    
    def get_node_by_attributes(self, **kwargs) -> List[ExplorationNode]:
        """Find nodes matching specific attributes"""
        matching_nodes = []
        for node in self.nodes.values():
            match = True
            for key, value in kwargs.items():
                if key not in node.attributes or node.attributes[key] != value:
                    match = False
                    break
            if match:
                matching_nodes.append(node)
        return matching_nodes
    
    def update_node_statistics(self, node_id: str, success: bool, 
                             reward: float, novel: bool = False) -> None:
        """Update node statistics after exploration"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.exploration_count += 1
        node.last_explored = datetime.now()
        
        # Update success rate using exponential moving average
        alpha = 0.3  # Learning rate
        if success:
            node.success_rate = alpha * 1.0 + (1 - alpha) * node.success_rate
        else:
            node.success_rate = alpha * 0.0 + (1 - alpha) * node.success_rate
        
        # Update reward potential
        node.reward_potential = alpha * reward + (1 - alpha) * node.reward_potential
        
        # Update novelty (decreases with exploration)
        if novel:
            node.novelty_score = min(1.0, node.novelty_score + 0.1)
            self.novel_discoveries += 1
        else:
            node.novelty_score = max(0.1, node.novelty_score * 0.95)
        
        # Update global statistics
        self.total_explorations += 1
        if success:
            self.successful_explorations += 1
        
        logger.debug(f"Updated node {node_id}: success_rate={node.success_rate:.3f}, "
                    f"novelty={node.novelty_score:.3f}")
    
    def update_edge_statistics(self, source_id: str, target_id: str, 
                             success: bool) -> None:
        """Update edge statistics after traversal"""
        
        edge = self._find_edge(source_id, target_id)
        if not edge:
            return
        
        edge.traversal_count += 1
        
        # Update success rate
        alpha = 0.3
        if success:
            edge.success_rate = alpha * 1.0 + (1 - alpha) * edge.success_rate
        else:
            edge.success_rate = alpha * 0.0 + (1 - alpha) * edge.success_rate
        
        # Update graph edge attributes
        self.graph[source_id][target_id]['success_rate'] = edge.success_rate
    
    def _find_edge(self, source_id: str, target_id: str) -> Optional[ExplorationEdge]:
        """Find edge between two nodes"""
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.node_type.value] += 1
        
        edge_types = defaultdict(int)
        for edge in self.edges:
            edge_types[edge.edge_type] += 1
        
        # Graph topology metrics
        if len(self.nodes) > 0:
            avg_degree = len(self.edges) / len(self.nodes)
            
            # Find connected components
            undirected_graph = self.graph.to_undirected()
            connected_components = nx.number_connected_components(undirected_graph)
            
            # Calculate clustering coefficient
            clustering = nx.average_clustering(undirected_graph)
        else:
            avg_degree = 0
            connected_components = 0
            clustering = 0
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'avg_degree': avg_degree,
            'connected_components': connected_components,
            'clustering_coefficient': clustering,
            'total_explorations': self.total_explorations,
            'success_rate': self.successful_explorations / max(1, self.total_explorations),
            'novel_discoveries': self.novel_discoveries
        }
    
    def evolve_graph(self) -> None:
        """Dynamically evolve the graph based on exploration results"""
        
        # Apply node creation rules
        for rule in self.node_creation_rules:
            new_nodes = rule(self)
            for node in new_nodes:
                self.add_node(node)
        
        # Apply edge creation rules
        for rule in self.edge_creation_rules:
            new_edges = rule(self)
            for edge in new_edges:
                self.add_edge(edge)
        
        logger.info(f"Graph evolution: {len(self.nodes)} nodes, {len(self.edges)} edges")


class PathFinder(ABC):
    """Abstract base class for pathfinding algorithms"""
    
    @abstractmethod
    async def find_path(self, graph: AttackSpaceGraph, start_node: str, 
                       end_node: Optional[str], constraints: Dict[str, Any]) -> List[str]:
        """Find path through the attack graph"""
        pass


class AdaptiveAStarPathfinder(PathFinder):
    """
    Adaptive A* pathfinding that learns from successful attack paths.
    Optimizes for different objectives based on current exploration strategy.
    """
    
    def __init__(self):
        self.path_memory: Dict[Tuple[str, str], List[str]] = {}
        self.heuristic_weights = {
            'distance': 0.4,
            'success_rate': 0.3,
            'novelty': 0.2,
            'difficulty': 0.1
        }
    
    async def find_path(self, graph: AttackSpaceGraph, start_node: str, 
                       end_node: Optional[str], constraints: Dict[str, Any]) -> List[str]:
        """Find optimal path using adaptive A*"""
        
        if start_node not in graph.nodes:
            logger.warning(f"Start node {start_node} not found in graph")
            return []
        
        strategy = constraints.get('strategy', 'balanced')
        max_depth = constraints.get('max_depth', 10)
        objective = constraints.get('objective', 'explore')
        
        # If no specific end node, find best exploration target
        if end_node is None:
            end_node = await self._find_exploration_target(graph, start_node, strategy, objective)
        
        if end_node is None or end_node not in graph.nodes:
            return [start_node]  # Stay at start if no target found
        
        # Check if we have a cached path
        cache_key = (start_node, end_node)
        if cache_key in self.path_memory and random.random() < 0.3:  # 30% chance to use cached path
            cached_path = self.path_memory[cache_key]
            if self._validate_path(graph, cached_path):
                logger.debug(f"Using cached path from {start_node} to {end_node}")
                return cached_path
        
        # Run A* algorithm
        path = await self._astar_search(graph, start_node, end_node, strategy, max_depth)
        
        # Cache successful paths
        if len(path) > 1:
            self.path_memory[cache_key] = path
            
            # Limit cache size
            if len(self.path_memory) > 100:
                # Remove oldest entries
                oldest_key = min(self.path_memory.keys())
                del self.path_memory[oldest_key]
        
        return path
    
    async def _find_exploration_target(self, graph: AttackSpaceGraph, start_node: str, 
                                     strategy: str, objective: str) -> Optional[str]:
        """Find best target node for exploration"""
        
        candidates = []
        
        if objective == 'explore':
            # Find nodes with high novelty or low exploration count
            for node_id, node in graph.nodes.items():
                if node_id != start_node:
                    score = node.get_priority_score('exploration')
                    candidates.append((node_id, score))
        
        elif objective == 'exploit':
            # Find nodes with high success rates
            for node_id, node in graph.nodes.items():
                if node_id != start_node and node.success_rate > 0.5:
                    score = node.get_priority_score('exploitation')
                    candidates.append((node_id, score))
        
        elif objective == 'diverse':
            # Find nodes that are topologically distant
            for node_id, node in graph.nodes.items():
                if node_id != start_node:
                    try:
                        distance = nx.shortest_path_length(graph.graph, start_node, node_id)
                        diversity_score = distance / 10.0  # Normalize
                        score = diversity_score * node.novelty_score
                        candidates.append((node_id, score))
                    except nx.NetworkXNoPath:
                        # No path exists, high diversity score
                        candidates.append((node_id, 1.0))
        
        else:  # balanced
            for node_id, node in graph.nodes.items():
                if node_id != start_node:
                    score = node.get_priority_score('balanced')
                    candidates.append((node_id, score))
        
        # Select target based on weighted random sampling
        if not candidates:
            return None
        
        # Sort by score and apply softmax-like selection
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:min(5, len(candidates))]  # Top 5 candidates
        
        # Weighted random selection
        total_weight = sum(score for _, score in top_candidates)
        if total_weight == 0:
            return random.choice(top_candidates)[0]
        
        r = random.random() * total_weight
        cumulative = 0
        for node_id, score in top_candidates:
            cumulative += score
            if r <= cumulative:
                return node_id
        
        return top_candidates[0][0]  # Fallback to best candidate
    
    async def _astar_search(self, graph: AttackSpaceGraph, start: str, goal: str, 
                          strategy: str, max_depth: int) -> List[str]:
        """A* search implementation with adaptive heuristics"""
        
        # Priority queue: (f_score, g_score, node_id, path)
        open_set = [(0, 0, start, [start])]
        closed_set = set()
        
        # Track best g_score for each node
        g_scores = {start: 0}
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current == goal:
                logger.debug(f"Found path from {start} to {goal}: {' -> '.join(path)}")
                return path
            
            if current in closed_set or len(path) > max_depth:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            neighbors = graph.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Calculate cost to reach neighbor
                edge = graph._find_edge(current, neighbor)
                if not edge:
                    continue
                
                tentative_g = g_score + self._calculate_edge_cost(edge, strategy)
                
                # Skip if we've found a better path to this neighbor
                if neighbor in g_scores and tentative_g >= g_scores[neighbor]:
                    continue
                
                g_scores[neighbor] = tentative_g
                h_score = self._heuristic(graph, neighbor, goal, strategy)
                f_score = tentative_g + h_score
                
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))
        
        logger.debug(f"No path found from {start} to {goal}")
        return [start]  # Return start node if no path found
    
    def _calculate_edge_cost(self, edge: ExplorationEdge, strategy: str) -> float:
        """Calculate cost of traversing an edge"""
        
        base_cost = edge.weight
        
        if strategy == 'conservative':
            # High cost for risky/untested edges
            if edge.traversal_count == 0:
                base_cost *= 2.0
            elif edge.success_rate < 0.3:
                base_cost *= 1.5
        
        elif strategy == 'aggressive':
            # Lower cost for high-risk, high-reward edges
            if edge.weight > 0.7:  # High-risk edge
                base_cost *= 0.7
        
        elif strategy == 'exploratory':
            # Lower cost for unexplored edges
            novelty_multiplier = 1.0 / (edge.traversal_count + 1)
            base_cost *= (1.0 - novelty_multiplier * 0.5)
        
        return max(0.1, base_cost)
    
    def _heuristic(self, graph: AttackSpaceGraph, current: str, goal: str, strategy: str) -> float:
        """Adaptive heuristic function"""
        
        if current == goal:
            return 0.0
        
        try:
            # Topological distance
            distance = nx.shortest_path_length(graph.graph.to_undirected(), current, goal)
            distance_component = distance * self.heuristic_weights['distance']
        except nx.NetworkXNoPath:
            distance_component = 10.0  # High cost for unreachable nodes
        
        # Node characteristics
        current_node = graph.nodes[current]
        goal_node = graph.nodes[goal]
        
        # Success rate component (prefer paths through successful nodes)
        success_component = (1.0 - current_node.success_rate) * self.heuristic_weights['success_rate']
        
        # Novelty component
        novelty_component = (1.0 - current_node.novelty_score) * self.heuristic_weights['novelty']
        
        # Difficulty component
        difficulty_component = current_node.difficulty * self.heuristic_weights['difficulty']
        
        total_heuristic = distance_component + success_component + novelty_component + difficulty_component
        
        # Strategy-specific adjustments
        if strategy == 'exploration':
            total_heuristic *= (2.0 - current_node.novelty_score)  # Prefer novel nodes
        elif strategy == 'exploitation':
            total_heuristic *= (2.0 - current_node.success_rate)   # Prefer successful nodes
        
        return max(0.1, total_heuristic)
    
    def _validate_path(self, graph: AttackSpaceGraph, path: List[str]) -> bool:
        """Validate that a cached path is still valid"""
        
        if len(path) < 2:
            return True
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if current not in graph.nodes or next_node not in graph.nodes:
                return False
            
            if not graph._find_edge(current, next_node):
                return False
        
        return True


class MultiObjectivePathfinder(PathFinder):
    """
    Multi-objective pathfinding that optimizes for multiple goals simultaneously.
    Uses Pareto-optimal solutions for complex attack planning.
    """
    
    def __init__(self):
        self.objectives = ['success', 'novelty', 'stealth', 'efficiency']
        self.pareto_solutions: List[Dict[str, Any]] = []
    
    async def find_path(self, graph: AttackSpaceGraph, start_node: str, 
                       end_node: Optional[str], constraints: Dict[str, Any]) -> List[str]:
        """Find Pareto-optimal path"""
        
        if end_node is None:
            # Find multiple potential targets and evaluate paths
            potential_targets = self._find_potential_targets(graph, start_node, constraints)
            
            if not potential_targets:
                return [start_node]
            
            # Evaluate paths to all targets
            path_solutions = []
            for target in potential_targets:
                path = await self._find_single_objective_path(graph, start_node, target, 'balanced')
                if len(path) > 1:
                    objectives_scores = self._evaluate_path_objectives(graph, path)
                    path_solutions.append({
                        'path': path,
                        'target': target,
                        'objectives': objectives_scores
                    })
            
            # Select path using Pareto dominance
            if path_solutions:
                pareto_paths = self._find_pareto_optimal_paths(path_solutions)
                
                # Select best path from Pareto set (could use additional criteria)
                best_path = self._select_from_pareto_set(pareto_paths, constraints)
                return best_path['path']
        
        else:
            # Single target - find best path
            return await self._find_single_objective_path(graph, start_node, end_node, 'balanced')
        
        return [start_node]
    
    def _find_potential_targets(self, graph: AttackSpaceGraph, start_node: str, 
                              constraints: Dict[str, Any]) -> List[str]:
        """Find potential target nodes for multi-objective optimization"""
        
        candidates = []
        min_novelty = constraints.get('min_novelty', 0.3)
        min_success_potential = constraints.get('min_success_potential', 0.2)
        
        for node_id, node in graph.nodes.items():
            if (node_id != start_node and 
                node.novelty_score >= min_novelty and 
                node.reward_potential >= min_success_potential):
                candidates.append(node_id)
        
        # Limit number of candidates for computational efficiency
        return candidates[:10]
    
    async def _find_single_objective_path(self, graph: AttackSpaceGraph, start: str, 
                                        end: str, strategy: str) -> List[str]:
        """Find path optimizing single objective (used as subroutine)"""
        
        # Use simplified Dijkstra's algorithm
        distances = {start: 0}
        previous = {}
        unvisited = set(graph.nodes.keys())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
            
            if distances.get(current, float('inf')) == float('inf'):
                break  # No more reachable nodes
            
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                return path[::-1]
            
            unvisited.remove(current)
            
            # Check neighbors
            neighbors = graph.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in unvisited:
                    edge = graph._find_edge(current, neighbor)
                    if edge:
                        alt_distance = distances[current] + edge.weight
                        if alt_distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = alt_distance
                            previous[neighbor] = current
        
        return [start]  # No path found
    
    def _evaluate_path_objectives(self, graph: AttackSpaceGraph, path: List[str]) -> Dict[str, float]:
        """Evaluate a path against multiple objectives"""
        
        if len(path) < 2:
            return {obj: 0.0 for obj in self.objectives}
        
        objectives = {}
        
        # Success objective: average success rate of nodes and edges
        node_success_rates = [graph.nodes[node_id].success_rate for node_id in path]
        edge_success_rates = []
        for i in range(len(path) - 1):
            edge = graph._find_edge(path[i], path[i + 1])
            if edge:
                edge_success_rates.append(edge.success_rate)
        
        avg_node_success = sum(node_success_rates) / len(node_success_rates)
        avg_edge_success = sum(edge_success_rates) / len(edge_success_rates) if edge_success_rates else 0
        objectives['success'] = (avg_node_success + avg_edge_success) / 2
        
        # Novelty objective: average novelty of nodes
        novelty_scores = [graph.nodes[node_id].novelty_score for node_id in path]
        objectives['novelty'] = sum(novelty_scores) / len(novelty_scores)
        
        # Stealth objective: inverse of difficulty
        difficulties = [graph.nodes[node_id].difficulty for node_id in path]
        objectives['stealth'] = 1.0 - (sum(difficulties) / len(difficulties))
        
        # Efficiency objective: inverse of path length and edge weights
        total_weight = 0
        for i in range(len(path) - 1):
            edge = graph._find_edge(path[i], path[i + 1])
            if edge:
                total_weight += edge.weight
        
        path_efficiency = 1.0 / (1.0 + total_weight + len(path) * 0.1)
        objectives['efficiency'] = path_efficiency
        
        return objectives
    
    def _find_pareto_optimal_paths(self, path_solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto-optimal solutions from path candidates"""
        
        pareto_optimal = []
        
        for i, solution_a in enumerate(path_solutions):
            is_dominated = False
            
            for j, solution_b in enumerate(path_solutions):
                if i != j and self._dominates(solution_b['objectives'], solution_a['objectives']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(solution_a)
        
        return pareto_optimal
    
    def _dominates(self, objectives_a: Dict[str, float], objectives_b: Dict[str, float]) -> bool:
        """Check if solution A dominates solution B (all objectives >= and at least one >)"""
        
        all_greater_equal = True
        at_least_one_greater = False
        
        for objective in self.objectives:
            a_val = objectives_a.get(objective, 0)
            b_val = objectives_b.get(objective, 0)
            
            if a_val < b_val:
                all_greater_equal = False
                break
            elif a_val > b_val:
                at_least_one_greater = True
        
        return all_greater_equal and at_least_one_greater
    
    def _select_from_pareto_set(self, pareto_paths: List[Dict[str, Any]], 
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select best path from Pareto-optimal set"""
        
        if not pareto_paths:
            return {'path': []}
        
        if len(pareto_paths) == 1:
            return pareto_paths[0]
        
        # Use weighted sum based on user preferences
        objective_weights = constraints.get('objective_weights', {
            'success': 0.4,
            'novelty': 0.3,
            'stealth': 0.2,
            'efficiency': 0.1
        })
        
        best_solution = None
        best_score = -1
        
        for solution in pareto_paths:
            weighted_score = sum(
                solution['objectives'].get(obj, 0) * objective_weights.get(obj, 0)
                for obj in self.objectives
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_solution = solution
        
        return best_solution or pareto_paths[0]


class DynamicExplorer:
    """
    Dynamic exploration engine that orchestrates graph-based attack space exploration.
    Adaptively selects pathfinding strategies and evolves the graph based on results.
    """
    
    def __init__(self, graph: AttackSpaceGraph):
        self.graph = graph
        self.pathfinders = {
            'adaptive_astar': AdaptiveAStarPathfinder(),
            'multi_objective': MultiObjectivePathfinder()
        }
        
        # Exploration strategy state
        self.current_strategy = 'balanced'
        self.strategy_performance = {
            'balanced': {'success_count': 0, 'total_count': 0},
            'exploration': {'success_count': 0, 'total_count': 0},
            'exploitation': {'success_count': 0, 'total_count': 0},
            'aggressive': {'success_count': 0, 'total_count': 0},
            'conservative': {'success_count': 0, 'total_count': 0}
        }
        
        # Adaptive parameters
        self.strategy_switch_threshold = 0.1  # Switch if performance drops below this
        self.exploration_bias = 0.6  # Bias towards exploration vs exploitation
        self.recent_results = deque(maxlen=20)  # Track recent exploration results
        
    async def explore_next(self, current_node: str, 
                          constraints: Dict[str, Any] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Find next exploration path and return path with metadata.
        
        Returns:
            Tuple of (path, exploration_metadata)
        """
        
        constraints = constraints or {}
        
        # Adapt strategy based on recent performance
        self._adapt_exploration_strategy()
        
        # Select pathfinder based on exploration goals
        pathfinder_name = self._select_pathfinder(constraints)
        pathfinder = self.pathfinders[pathfinder_name]
        
        # Set strategy-specific constraints
        exploration_constraints = self._prepare_constraints(constraints)
        
        # Find path
        logger.info(f"Exploring from {current_node} using {pathfinder_name} with strategy {self.current_strategy}")
        
        path = await pathfinder.find_path(
            self.graph, 
            current_node, 
            constraints.get('target_node'),
            exploration_constraints
        )
        
        # Prepare exploration metadata
        metadata = {
            'pathfinder_used': pathfinder_name,
            'strategy': self.current_strategy,
            'path_length': len(path),
            'estimated_difficulty': self._estimate_path_difficulty(path),
            'estimated_reward': self._estimate_path_reward(path),
            'novelty_score': self._estimate_path_novelty(path)
        }
        
        return path, metadata
    
    def report_exploration_result(self, path: List[str], success: bool, 
                                reward: float, novel_discoveries: int = 0) -> None:
        """Report the result of an exploration for adaptive learning"""
        
        # Update strategy performance
        self.strategy_performance[self.current_strategy]['total_count'] += 1
        if success:
            self.strategy_performance[self.current_strategy]['success_count'] += 1
        
        # Update graph statistics
        for i, node_id in enumerate(path):
            is_novel = novel_discoveries > 0 and i < novel_discoveries
            self.graph.update_node_statistics(node_id, success, reward, is_novel)
        
        # Update edge statistics
        for i in range(len(path) - 1):
            self.graph.update_edge_statistics(path[i], path[i + 1], success)
        
        # Track recent results for adaptation
        self.recent_results.append({
            'success': success,
            'reward': reward,
            'strategy': self.current_strategy,
            'path_length': len(path),
            'novel_discoveries': novel_discoveries
        })
        
        logger.info(f"Exploration result: success={success}, reward={reward:.3f}, "
                   f"novel_discoveries={novel_discoveries}")
        
        # Evolve graph based on results
        if novel_discoveries > 0 or reward > 0.7:
            self.graph.evolve_graph()
    
    def _adapt_exploration_strategy(self) -> None:
        """Adapt exploration strategy based on recent performance"""
        
        if len(self.recent_results) < 5:
            return  # Not enough data
        
        # Calculate recent success rate
        recent_successes = sum(1 for result in self.recent_results if result['success'])
        recent_success_rate = recent_successes / len(self.recent_results)
        
        # Calculate average reward
        recent_rewards = [result['reward'] for result in self.recent_results]
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Calculate novelty rate
        novel_count = sum(result['novel_discoveries'] for result in self.recent_results)
        novelty_rate = novel_count / len(self.recent_results)
        
        # Strategy switching logic
        current_performance = self.strategy_performance[self.current_strategy]
        current_success_rate = (current_performance['success_count'] / 
                               max(1, current_performance['total_count']))
        
        # Switch strategy if performance is poor
        if current_success_rate < self.strategy_switch_threshold:
            logger.info(f"Switching from {self.current_strategy} due to poor performance: "
                       f"{current_success_rate:.3f}")
            self._select_new_strategy(recent_success_rate, avg_recent_reward, novelty_rate)
        
        # Adjust exploration bias
        if novelty_rate < 0.1 and recent_success_rate > 0.6:
            # Low novelty but good success - increase exploration
            self.exploration_bias = min(0.8, self.exploration_bias + 0.1)
        elif novelty_rate > 0.5 and recent_success_rate < 0.3:
            # High novelty but poor success - increase exploitation
            self.exploration_bias = max(0.2, self.exploration_bias - 0.1)
    
    def _select_new_strategy(self, recent_success_rate: float, 
                           avg_reward: float, novelty_rate: float) -> None:
        """Select new exploration strategy based on current state"""
        
        if recent_success_rate < 0.3:
            # Poor success rate - try conservative approach
            new_strategy = 'conservative'
        elif novelty_rate < 0.1:
            # Low novelty - increase exploration
            new_strategy = 'exploration'
        elif avg_reward > 0.7:
            # High rewards - try aggressive approach
            new_strategy = 'aggressive'
        else:
            # Default to balanced
            new_strategy = 'balanced'
        
        if new_strategy != self.current_strategy:
            logger.info(f"Strategy changed from {self.current_strategy} to {new_strategy}")
            self.current_strategy = new_strategy
    
    def _select_pathfinder(self, constraints: Dict[str, Any]) -> str:
        """Select appropriate pathfinder based on constraints and state"""
        
        # Multi-objective pathfinding for complex scenarios
        if ('objective_weights' in constraints or 
            constraints.get('multi_objective', False) or
            len(self.recent_results) > 10):  # Use multi-objective when we have enough data
            return 'multi_objective'
        
        # Default to adaptive A*
        return 'adaptive_astar'
    
    def _prepare_constraints(self, base_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare exploration constraints based on current strategy"""
        
        constraints = base_constraints.copy()
        constraints['strategy'] = self.current_strategy
        
        # Strategy-specific parameters
        if self.current_strategy == 'exploration':
            constraints['objective'] = 'explore'
            constraints['max_depth'] = constraints.get('max_depth', 8)
        elif self.current_strategy == 'exploitation':
            constraints['objective'] = 'exploit'
            constraints['max_depth'] = constraints.get('max_depth', 5)
        elif self.current_strategy == 'aggressive':
            constraints['objective'] = 'diverse'
            constraints['max_depth'] = constraints.get('max_depth', 12)
        else:  # balanced, conservative
            constraints['objective'] = 'balanced'
            constraints['max_depth'] = constraints.get('max_depth', 6)
        
        return constraints
    
    def _estimate_path_difficulty(self, path: List[str]) -> float:
        """Estimate overall difficulty of a path"""
        if len(path) < 2:
            return 0.0
        
        difficulties = []
        for node_id in path:
            if node_id in self.graph.nodes:
                difficulties.append(self.graph.nodes[node_id].difficulty)
        
        # Weight more heavily the most difficult nodes
        if difficulties:
            difficulties.sort(reverse=True)
            weighted_difficulty = sum(
                diff * (0.8 ** i) for i, diff in enumerate(difficulties)
            ) / sum(0.8 ** i for i in range(len(difficulties)))
            return weighted_difficulty
        
        return 0.5
    
    def _estimate_path_reward(self, path: List[str]) -> float:
        """Estimate potential reward of a path"""
        if len(path) < 2:
            return 0.0
        
        rewards = []
        for node_id in path:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                # Combine reward potential with success rate
                estimated_reward = node.reward_potential * node.success_rate
                rewards.append(estimated_reward)
        
        if rewards:
            # Use max reward as the potential (optimistic estimate)
            return max(rewards)
        
        return 0.0
    
    def _estimate_path_novelty(self, path: List[str]) -> float:
        """Estimate novelty of a path"""
        if len(path) < 2:
            return 0.0
        
        novelties = []
        for node_id in path:
            if node_id in self.graph.nodes:
                novelties.append(self.graph.nodes[node_id].novelty_score)
        
        if novelties:
            # Average novelty of path
            return sum(novelties) / len(novelties)
        
        return 0.0
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive exploration statistics"""
        
        graph_stats = self.graph.get_graph_statistics()
        
        strategy_stats = {}
        for strategy, performance in self.strategy_performance.items():
            total = performance['total_count']
            success = performance['success_count']
            strategy_stats[strategy] = {
                'total_attempts': total,
                'success_rate': success / max(1, total),
                'success_count': success
            }
        
        recent_stats = {}
        if self.recent_results:
            recent_stats = {
                'recent_success_rate': sum(1 for r in self.recent_results if r['success']) / len(self.recent_results),
                'recent_avg_reward': sum(r['reward'] for r in self.recent_results) / len(self.recent_results),
                'recent_novelty_rate': sum(r['novel_discoveries'] for r in self.recent_results) / len(self.recent_results),
                'avg_path_length': sum(r['path_length'] for r in self.recent_results) / len(self.recent_results)
            }
        
        return {
            'current_strategy': self.current_strategy,
            'exploration_bias': self.exploration_bias,
            'graph_statistics': graph_stats,
            'strategy_performance': strategy_stats,
            'recent_performance': recent_stats,
            'pathfinder_capabilities': list(self.pathfinders.keys())
        }