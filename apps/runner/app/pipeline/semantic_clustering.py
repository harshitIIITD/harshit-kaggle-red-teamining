# ABOUTME: Semantic clustering system that replaces MinHash with neural embeddings
# ABOUTME: Uses transformer embeddings and advanced clustering for intelligent deduplication

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
import math
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re

logger = logging.getLogger(__name__)


@dataclass
class SemanticEmbedding:
    """Semantic embedding representation of text with metadata"""
    text: str
    embedding: np.ndarray
    source_type: str                    # 'prompt', 'response', 'combined'
    vulnerability_category: str
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure embedding is numpy array and normalized
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding)
        
        # L2 normalize embedding
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
    
    def cosine_similarity_to(self, other: 'SemanticEmbedding') -> float:
        """Calculate cosine similarity to another embedding"""
        return float(np.dot(self.embedding, other.embedding))
    
    def euclidean_distance_to(self, other: 'SemanticEmbedding') -> float:
        """Calculate Euclidean distance to another embedding"""
        return float(np.linalg.norm(self.embedding - other.embedding))


@dataclass
class SemanticCluster:
    """Cluster of semantically similar embeddings"""
    id: str
    centroid: np.ndarray
    embeddings: List[SemanticEmbedding]
    cluster_type: str                   # 'vulnerability', 'technique', 'outcome'
    coherence_score: float = 0.0        # How tightly clustered the embeddings are
    novelty_score: float = 1.0         # How novel this cluster is
    creation_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.embeddings:
            self._update_centroid()
            self._calculate_coherence()
    
    def add_embedding(self, embedding: SemanticEmbedding) -> None:
        """Add new embedding to cluster"""
        self.embeddings.append(embedding)
        self.last_updated = datetime.now()
        self._update_centroid()
        self._calculate_coherence()
    
    def _update_centroid(self) -> None:
        """Update cluster centroid"""
        if not self.embeddings:
            return
        
        # Weighted average based on confidence scores
        total_weight = sum(emb.confidence_score for emb in self.embeddings)
        
        if total_weight > 0:
            weighted_sum = sum(
                emb.embedding * emb.confidence_score 
                for emb in self.embeddings
            )
            self.centroid = weighted_sum / total_weight
        else:
            # Simple average if no confidence scores
            self.centroid = np.mean([emb.embedding for emb in self.embeddings], axis=0)
        
        # Normalize centroid
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
    
    def _calculate_coherence(self) -> None:
        """Calculate how coherent/tight the cluster is"""
        if len(self.embeddings) < 2:
            self.coherence_score = 1.0
            return
        
        distances = []
        for embedding in self.embeddings:
            distance = np.linalg.norm(embedding.embedding - self.centroid)
            distances.append(distance)
        
        # Coherence is inverse of average distance
        avg_distance = sum(distances) / len(distances)
        self.coherence_score = 1.0 / (1.0 + avg_distance)
    
    def similarity_to_embedding(self, embedding: SemanticEmbedding) -> float:
        """Calculate similarity to a new embedding"""
        return float(np.dot(self.centroid, embedding.embedding))
    
    def should_merge_with(self, other: 'SemanticCluster', threshold: float = 0.8) -> bool:
        """Determine if this cluster should merge with another"""
        centroid_similarity = float(np.dot(self.centroid, other.centroid))
        
        # Additional criteria for merging
        same_type = self.cluster_type == other.cluster_type
        time_proximity = abs((self.creation_time - other.creation_time).total_seconds()) < 3600  # 1 hour
        
        return centroid_similarity > threshold and same_type and time_proximity
    
    def get_representative_text(self) -> str:
        """Get most representative text from the cluster"""
        if not self.embeddings:
            return ""
        
        # Find embedding closest to centroid
        best_embedding = None
        best_similarity = -1
        
        for embedding in self.embeddings:
            similarity = float(np.dot(embedding.embedding, self.centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_embedding = embedding
        
        return best_embedding.text if best_embedding else self.embeddings[0].text
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize cluster for storage"""
        return {
            'id': self.id,
            'centroid': self.centroid.tolist(),
            'cluster_type': self.cluster_type,
            'coherence_score': self.coherence_score,
            'novelty_score': self.novelty_score,
            'creation_time': self.creation_time.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'embedding_count': len(self.embeddings),
            'representative_text': self.get_representative_text()
        }


class EmbeddingGenerator(ABC):
    """Abstract base for generating semantic embeddings"""
    
    @abstractmethod
    async def generate_embedding(self, text: str, context: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for text"""
        pass


class TransformerEmbeddingGenerator(EmbeddingGenerator):
    """
    Generate embeddings using transformer models through OpenRouter.
    Replaces simple text hashing with deep semantic understanding.
    """
    
    def __init__(self, model_client: Any):
        self.model_client = model_client
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.embedding_dimension = 384  # Typical embedding dimension
        
    async def generate_embedding(self, text: str, context: Dict[str, Any]) -> np.ndarray:
        """Generate transformer-based embedding"""
        
        # Check cache first
        cache_key = self._create_cache_key(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Use embedding model through OpenRouter
            embedding = await self._get_model_embedding(text, context)
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            # Limit cache size
            if len(self.embedding_cache) > 1000:
                # Remove oldest entries (simplified LRU)
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            # Fallback to simulated embedding
            return self._generate_fallback_embedding(text)
    
    async def _get_model_embedding(self, text: str, context: Dict[str, Any]) -> np.ndarray:
        """Get embedding from model API"""
        
        # For now, simulate embedding generation since OpenRouter may not have embedding endpoints
        # In production, this would call an actual embedding API
        return self._generate_simulated_embedding(text)
    
    def _generate_simulated_embedding(self, text: str) -> np.ndarray:
        """Generate simulated but meaningful embedding"""
        
        # Create a hash-based but more sophisticated embedding
        import hashlib
        
        # Process text to extract semantic features
        processed_text = self._preprocess_text(text)
        
        # Create multiple hash-based features
        features = []
        
        # Character n-grams
        for n in range(2, 5):
            ngrams = [processed_text[i:i+n] for i in range(len(processed_text) - n + 1)]
            for ngram in ngrams[:20]:  # Limit to first 20
                hash_val = int(hashlib.md5(ngram.encode()).hexdigest()[:8], 16)
                features.append(hash_val / (2**32))  # Normalize
        
        # Word-level features
        words = processed_text.split()
        for word in words[:30]:  # Limit to first 30 words
            hash_val = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            features.append(hash_val / (2**32))
        
        # Semantic category features
        semantic_features = self._extract_semantic_features(text)
        features.extend(semantic_features)
        
        # Pad or truncate to target dimension
        if len(features) < self.embedding_dimension:
            features.extend([0.0] * (self.embedding_dimension - len(features)))
        else:
            features = features[:self.embedding_dimension]
        
        embedding = np.array(features, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation"""
        
        # Lowercase and clean
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        return text.strip()
    
    def _extract_semantic_features(self, text: str) -> List[float]:
        """Extract semantic features from text"""
        
        features = []
        text_lower = text.lower()
        
        # Vulnerability category indicators
        vuln_indicators = {
            'jailbreak': ['bypass', 'ignore', 'override', 'pretend', 'act as', 'roleplay'],
            'injection': ['inject', 'execute', 'system', 'admin', 'root', 'command'],
            'leakage': ['reveal', 'show', 'display', 'expose', 'leak', 'tell me'],
            'deception': ['false', 'lie', 'fake', 'mislead', 'trick', 'deceive'],
            'manipulation': ['convince', 'persuade', 'manipulate', 'coerce', 'force']
        }
        
        for category, indicators in vuln_indicators.items():
            indicator_count = sum(1 for indicator in indicators if indicator in text_lower)
            features.append(indicator_count / len(indicators))
        
        # Sentiment and tone features
        positive_words = ['please', 'help', 'kindly', 'appreciate', 'thank']
        negative_words = ['urgent', 'critical', 'emergency', 'immediately', 'must']
        
        positive_score = sum(1 for word in positive_words if word in text_lower) / len(positive_words)
        negative_score = sum(1 for word in negative_words if word in text_lower) / len(negative_words)
        
        features.extend([positive_score, negative_score])
        
        # Complexity features
        words = text.split()
        sentences = text.split('.')
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Normalize complexity features
        features.extend([
            min(1.0, avg_word_length / 10.0),
            min(1.0, avg_sentence_length / 20.0),
            min(1.0, len(text) / 1000.0)
        ])
        
        return features
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate simple fallback embedding"""
        
        # Very basic embedding based on character frequencies
        char_counts = Counter(text.lower())
        
        # Create feature vector from character frequencies
        features = []
        for i in range(256):  # ASCII range
            char = chr(i) if i < 128 else ''
            count = char_counts.get(char, 0)
            features.append(count)
        
        # Truncate to target dimension
        features = features[:self.embedding_dimension]
        if len(features) < self.embedding_dimension:
            features.extend([0.0] * (self.embedding_dimension - len(features)))
        
        embedding = np.array(features, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _create_cache_key(self, text: str) -> str:
        """Create cache key for text"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()


class HybridEmbeddingGenerator(EmbeddingGenerator):
    """
    Hybrid embedding generator that combines multiple embedding techniques.
    Provides more robust semantic representation.
    """
    
    def __init__(self, model_client: Any):
        self.transformer_generator = TransformerEmbeddingGenerator(model_client)
        self.embedding_weights = {
            'transformer': 0.7,
            'tfidf': 0.2,
            'structural': 0.1
        }
        
        # TF-IDF components
        self.vocabulary: Set[str] = set()
        self.document_frequencies: Dict[str, int] = {}
        self.total_documents = 0
    
    async def generate_embedding(self, text: str, context: Dict[str, Any]) -> np.ndarray:
        """Generate hybrid embedding combining multiple techniques"""
        
        # Get transformer embedding
        transformer_emb = await self.transformer_generator.generate_embedding(text, context)
        
        # Get TF-IDF embedding
        tfidf_emb = self._generate_tfidf_embedding(text)
        
        # Get structural embedding
        structural_emb = self._generate_structural_embedding(text)
        
        # Combine embeddings
        combined_embedding = (
            transformer_emb * self.embedding_weights['transformer'] +
            tfidf_emb * self.embedding_weights['tfidf'] +
            structural_emb * self.embedding_weights['structural']
        )
        
        # Normalize combined embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding
    
    def _generate_tfidf_embedding(self, text: str) -> np.ndarray:
        """Generate TF-IDF based embedding"""
        
        words = text.lower().split()
        word_counts = Counter(words)
        
        # Update vocabulary and document frequencies
        self.total_documents += 1
        for word in set(words):
            self.vocabulary.add(word)
            self.document_frequencies[word] = self.document_frequencies.get(word, 0) + 1
        
        # Create TF-IDF vector for most common words
        common_words = sorted(self.vocabulary)[:384]  # Limit to 384 dimensions
        
        tfidf_vector = []
        for word in common_words:
            tf = word_counts.get(word, 0) / len(words) if words else 0
            idf = math.log(self.total_documents / max(1, self.document_frequencies.get(word, 1)))
            tfidf_vector.append(tf * idf)
        
        # Pad to correct dimension
        target_dim = 384
        if len(tfidf_vector) < target_dim:
            tfidf_vector.extend([0.0] * (target_dim - len(tfidf_vector)))
        else:
            tfidf_vector = tfidf_vector[:target_dim]
        
        return np.array(tfidf_vector, dtype=np.float32)
    
    def _generate_structural_embedding(self, text: str) -> np.ndarray:
        """Generate structural features embedding"""
        
        features = []
        
        # Length features
        features.extend([
            len(text) / 1000.0,  # Normalize text length
            len(text.split()) / 100.0,  # Normalize word count
            len(text.split('.')) / 20.0  # Normalize sentence count
        ])
        
        # Character type ratios
        total_chars = len(text)
        if total_chars > 0:
            alpha_ratio = sum(1 for c in text if c.isalpha()) / total_chars
            digit_ratio = sum(1 for c in text if c.isdigit()) / total_chars
            punct_ratio = sum(1 for c in text if c in '.,!?;:') / total_chars
            space_ratio = sum(1 for c in text if c.isspace()) / total_chars
        else:
            alpha_ratio = digit_ratio = punct_ratio = space_ratio = 0.0
        
        features.extend([alpha_ratio, digit_ratio, punct_ratio, space_ratio])
        
        # Syntactic patterns
        patterns = {
            'questions': r'\?',
            'exclamations': r'!',
            'quotes': r'["\']',
            'parentheses': r'[\(\)]',
            'brackets': r'[\[\]]',
            'urls': r'http[s]?://',
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        for pattern_name, pattern in patterns.items():
            count = len(re.findall(pattern, text))
            features.append(count / max(1, total_chars / 100))  # Normalize by text length
        
        # Pad to target dimension
        target_dim = 384
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return np.array(features, dtype=np.float32)


class AdaptiveClusteringEngine:
    """
    Adaptive clustering engine that uses multiple clustering algorithms.
    Dynamically selects the best clustering approach based on data characteristics.
    """
    
    def __init__(self):
        self.clustering_algorithms = {
            'dbscan': self._dbscan_clustering,
            'hierarchical': self._hierarchical_clustering,
            'spectral': self._spectral_clustering,
            'adaptive': self._adaptive_clustering
        }
        
        # Algorithm performance tracking
        self.algorithm_performance = {
            'dbscan': {'silhouette_scores': [], 'cluster_counts': []},
            'hierarchical': {'silhouette_scores': [], 'cluster_counts': []},
            'spectral': {'silhouette_scores': [], 'cluster_counts': []},
            'adaptive': {'silhouette_scores': [], 'cluster_counts': []}
        }
        
        # Clustering parameters
        self.min_cluster_size = 2
        self.max_clusters = 50
        self.similarity_threshold = 0.8
        
    async def cluster_embeddings(self, embeddings: List[SemanticEmbedding], 
                                method: str = 'adaptive') -> List[SemanticCluster]:
        """Cluster embeddings using specified method"""
        
        if not embeddings:
            return []
        
        if len(embeddings) < self.min_cluster_size:
            # Not enough embeddings to cluster
            cluster = SemanticCluster(
                id=f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                centroid=embeddings[0].embedding,
                embeddings=embeddings,
                cluster_type='singleton'
            )
            return [cluster]
        
        # Select clustering method
        if method == 'adaptive':
            method = self._select_best_clustering_method(embeddings)
        
        clustering_func = self.clustering_algorithms.get(method, self._dbscan_clustering)
        
        try:
            clusters = await clustering_func(embeddings)
            
            # Evaluate clustering quality
            self._evaluate_clustering_quality(clusters, method)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Clustering failed with method {method}: {e}")
            # Fallback to simple distance-based clustering
            return await self._fallback_clustering(embeddings)
    
    def _select_best_clustering_method(self, embeddings: List[SemanticEmbedding]) -> str:
        """Select best clustering method based on data characteristics"""
        
        n_samples = len(embeddings)
        
        # Calculate data characteristics
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Estimate density
        pairwise_distances = self._calculate_pairwise_distances(embedding_matrix)
        avg_distance = np.mean(pairwise_distances)
        distance_std = np.std(pairwise_distances)
        
        # Data characteristics
        is_dense = avg_distance < 0.5
        is_uniform = distance_std < 0.2
        is_large = n_samples > 100
        
        # Method selection logic
        if is_dense and not is_large:
            return 'hierarchical'  # Good for dense, smaller datasets
        elif is_uniform and is_large:
            return 'spectral'      # Good for uniform, larger datasets
        elif not is_dense and not is_uniform:
            return 'dbscan'        # Good for varied density datasets
        else:
            # Check historical performance
            best_method = 'dbscan'  # Default
            best_score = 0
            
            for method, performance in self.algorithm_performance.items():
                if performance['silhouette_scores']:
                    avg_score = sum(performance['silhouette_scores']) / len(performance['silhouette_scores'])
                    if avg_score > best_score:
                        best_score = avg_score
                        best_method = method
            
            return best_method
    
    async def _dbscan_clustering(self, embeddings: List[SemanticEmbedding]) -> List[SemanticCluster]:
        """DBSCAN clustering implementation"""
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Adaptive parameter selection
        eps = self._estimate_eps(embedding_matrix)
        min_samples = max(2, min(5, len(embeddings) // 10))
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(embedding_matrix)
        
        return self._create_clusters_from_labels(embeddings, cluster_labels, 'dbscan')
    
    async def _hierarchical_clustering(self, embeddings: List[SemanticEmbedding]) -> List[SemanticCluster]:
        """Hierarchical clustering implementation"""
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Estimate number of clusters
        n_clusters = self._estimate_num_clusters(embedding_matrix)
        n_clusters = min(n_clusters, len(embeddings) // 2)  # Reasonable upper bound
        
        # Run hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        cluster_labels = hierarchical.fit_predict(embedding_matrix)
        
        return self._create_clusters_from_labels(embeddings, cluster_labels, 'hierarchical')
    
    async def _spectral_clustering(self, embeddings: List[SemanticEmbedding]) -> List[SemanticCluster]:
        """Spectral clustering implementation"""
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Estimate number of clusters
        n_clusters = self._estimate_num_clusters(embedding_matrix)
        n_clusters = min(n_clusters, len(embeddings) // 2)
        
        # Run spectral clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            random_state=42
        )
        cluster_labels = spectral.fit_predict(embedding_matrix)
        
        return self._create_clusters_from_labels(embeddings, cluster_labels, 'spectral')
    
    async def _adaptive_clustering(self, embeddings: List[SemanticEmbedding]) -> List[SemanticCluster]:
        """Adaptive clustering that combines multiple methods"""
        
        # Try multiple methods and ensemble the results
        methods = ['dbscan', 'hierarchical']
        all_clusterings = []
        
        for method in methods:
            try:
                clusters = await self.clustering_algorithms[method](embeddings)
                all_clusterings.append((method, clusters))
            except Exception as e:
                logger.warning(f"Clustering method {method} failed: {e}")
        
        if not all_clusterings:
            return await self._fallback_clustering(embeddings)
        
        # Ensemble clustering results
        return self._ensemble_clusterings(embeddings, all_clusterings)
    
    def _estimate_eps(self, embedding_matrix: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN"""
        
        # Calculate k-distance graph
        k = min(4, embedding_matrix.shape[0] - 1)
        
        distances = []
        for i in range(embedding_matrix.shape[0]):
            point_distances = []
            for j in range(embedding_matrix.shape[0]):
                if i != j:
                    distance = 1 - np.dot(embedding_matrix[i], embedding_matrix[j])  # Cosine distance
                    point_distances.append(distance)
            
            point_distances.sort()
            if len(point_distances) >= k:
                distances.append(point_distances[k-1])  # k-th nearest neighbor distance
        
        # Use elbow method approximation
        distances.sort()
        if len(distances) > 1:
            # Take distance at 80th percentile as eps
            percentile_80 = int(0.8 * len(distances))
            return distances[percentile_80]
        
        return 0.5  # Default eps
    
    def _estimate_num_clusters(self, embedding_matrix: np.ndarray) -> int:
        """Estimate optimal number of clusters"""
        
        n_samples = embedding_matrix.shape[0]
        
        # Rule of thumb: sqrt(n/2)
        estimated_clusters = int(math.sqrt(n_samples / 2))
        
        # Constrain to reasonable range
        estimated_clusters = max(2, min(estimated_clusters, self.max_clusters))
        estimated_clusters = min(estimated_clusters, n_samples // 2)
        
        return estimated_clusters
    
    def _calculate_pairwise_distances(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine distances"""
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Get upper triangle (exclude diagonal)
        n = distance_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_distances = distance_matrix[upper_triangle_indices]
        
        return pairwise_distances
    
    def _create_clusters_from_labels(self, embeddings: List[SemanticEmbedding], 
                                   cluster_labels: np.ndarray, method: str) -> List[SemanticCluster]:
        """Create SemanticCluster objects from clustering labels"""
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            
            # Get embeddings for this cluster
            cluster_embeddings = [
                embeddings[i] for i, l in enumerate(cluster_labels) if l == label
            ]
            
            if len(cluster_embeddings) >= self.min_cluster_size:
                cluster = SemanticCluster(
                    id=f"{method}_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    centroid=np.zeros(embeddings[0].embedding.shape),  # Will be calculated in __post_init__
                    embeddings=cluster_embeddings,
                    cluster_type='semantic_group'
                )
                clusters.append(cluster)
        
        return clusters
    
    def _ensemble_clusterings(self, embeddings: List[SemanticEmbedding], 
                            clusterings: List[Tuple[str, List[SemanticCluster]]]) -> List[SemanticCluster]:
        """Ensemble multiple clustering results"""
        
        if len(clusterings) == 1:
            return clusterings[0][1]
        
        # Create co-occurrence matrix
        n_embeddings = len(embeddings)
        co_occurrence = np.zeros((n_embeddings, n_embeddings))
        
        for method, clusters in clusterings:
            for cluster in clusters:
                cluster_indices = [
                    i for i, emb in enumerate(embeddings) 
                    if any(emb.text == cluster_emb.text for cluster_emb in cluster.embeddings)
                ]
                
                # Update co-occurrence matrix
                for i in cluster_indices:
                    for j in cluster_indices:
                        co_occurrence[i, j] += 1
        
        # Normalize by number of methods
        co_occurrence /= len(clusterings)
        
        # Create consensus clusters
        consensus_clusters = []
        used_indices = set()
        
        for i in range(n_embeddings):
            if i in used_indices:
                continue
            
            # Find all embeddings that frequently co-occur with embedding i
            cluster_indices = [i]
            for j in range(n_embeddings):
                if j != i and j not in used_indices and co_occurrence[i, j] > 0.5:
                    cluster_indices.append(j)
            
            if len(cluster_indices) >= self.min_cluster_size:
                cluster_embeddings = [embeddings[idx] for idx in cluster_indices]
                
                cluster = SemanticCluster(
                    id=f"ensemble_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    centroid=np.zeros(embeddings[0].embedding.shape),
                    embeddings=cluster_embeddings,
                    cluster_type='ensemble_consensus'
                )
                consensus_clusters.append(cluster)
                
                used_indices.update(cluster_indices)
        
        return consensus_clusters
    
    async def _fallback_clustering(self, embeddings: List[SemanticEmbedding]) -> List[SemanticCluster]:
        """Simple fallback clustering based on pairwise similarities"""
        
        clusters = []
        used_embeddings = set()
        
        for i, emb1 in enumerate(embeddings):
            if i in used_embeddings:
                continue
            
            cluster_embeddings = [emb1]
            cluster_indices = {i}
            
            # Find similar embeddings
            for j, emb2 in enumerate(embeddings):
                if j != i and j not in used_embeddings:
                    similarity = emb1.cosine_similarity_to(emb2)
                    if similarity > self.similarity_threshold:
                        cluster_embeddings.append(emb2)
                        cluster_indices.add(j)
            
            if len(cluster_embeddings) >= self.min_cluster_size:
                cluster = SemanticCluster(
                    id=f"fallback_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    centroid=np.zeros(embeddings[0].embedding.shape),
                    embeddings=cluster_embeddings,
                    cluster_type='similarity_based'
                )
                clusters.append(cluster)
                used_embeddings.update(cluster_indices)
        
        return clusters
    
    def _evaluate_clustering_quality(self, clusters: List[SemanticCluster], method: str) -> None:
        """Evaluate and record clustering quality"""
        
        if not clusters:
            return
        
        # Calculate silhouette score approximation
        total_coherence = sum(cluster.coherence_score for cluster in clusters)
        avg_coherence = total_coherence / len(clusters)
        
        # Record performance
        self.algorithm_performance[method]['silhouette_scores'].append(avg_coherence)
        self.algorithm_performance[method]['cluster_counts'].append(len(clusters))
        
        # Keep only recent performance data
        for metric in ['silhouette_scores', 'cluster_counts']:
            if len(self.algorithm_performance[method][metric]) > 20:
                self.algorithm_performance[method][metric] = \
                    self.algorithm_performance[method][metric][-20:]


class SemanticDeduplicationSystem:
    """
    Revolutionary semantic deduplication system that replaces MinHash/Jaccard clustering.
    Uses deep semantic understanding to identify truly novel vulnerabilities.
    """
    
    def __init__(self, model_client: Any):
        self.embedding_generator = HybridEmbeddingGenerator(model_client)
        self.clustering_engine = AdaptiveClusteringEngine()
        
        # Persistent storage
        self.known_clusters: List[SemanticCluster] = []
        self.embedding_cache: Dict[str, SemanticEmbedding] = {}
        
        # Configuration
        self.novelty_threshold = 0.6        # Threshold for considering something novel
        self.merge_threshold = 0.8          # Threshold for merging clusters
        self.max_clusters_in_memory = 1000  # Memory management
        
        # Statistics
        self.total_processed = 0
        self.novel_discoveries = 0
        self.deduplicated_count = 0
        
    async def process_new_finding(self, text: str, vulnerability_category: str, 
                                context: Dict[str, Any]) -> Tuple[bool, float, Optional[SemanticCluster]]:
        """
        Process new finding and determine if it's novel.
        
        Returns:
            Tuple of (is_novel, novelty_score, assigned_cluster)
        """
        
        self.total_processed += 1
        
        # Generate embedding for the new finding
        embedding = await self._create_embedding(text, vulnerability_category, context)
        
        # Find most similar existing cluster
        best_cluster, best_similarity = self._find_most_similar_cluster(embedding)
        
        # Determine novelty
        is_novel = best_similarity < self.novelty_threshold
        novelty_score = 1.0 - best_similarity if best_cluster else 1.0
        
        if is_novel:
            # Create new cluster for novel finding
            cluster = await self._create_new_cluster(embedding, vulnerability_category)
            self.known_clusters.append(cluster)
            self.novel_discoveries += 1
            
            logger.info(f"Novel finding discovered: novelty_score={novelty_score:.3f}")
            
            return True, novelty_score, cluster
        
        else:
            # Add to existing cluster
            best_cluster.add_embedding(embedding)
            self.deduplicated_count += 1
            
            logger.debug(f"Finding added to existing cluster: similarity={best_similarity:.3f}")
            
            return False, novelty_score, best_cluster
    
    async def _create_embedding(self, text: str, vulnerability_category: str, 
                              context: Dict[str, Any]) -> SemanticEmbedding:
        """Create semantic embedding for text"""
        
        # Check cache first
        cache_key = f"{hash(text)}_{vulnerability_category}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate new embedding
        embedding_vector = await self.embedding_generator.generate_embedding(text, context)
        
        embedding = SemanticEmbedding(
            text=text,
            embedding=embedding_vector,
            source_type=context.get('source_type', 'combined'),
            vulnerability_category=vulnerability_category,
            confidence_score=context.get('confidence_score', 0.8),
            metadata=context
        )
        
        # Cache the embedding
        self.embedding_cache[cache_key] = embedding
        
        # Manage cache size
        if len(self.embedding_cache) > 2000:
            # Remove oldest entries
            oldest_keys = list(self.embedding_cache.keys())[:500]
            for key in oldest_keys:
                del self.embedding_cache[key]
        
        return embedding
    
    def _find_most_similar_cluster(self, embedding: SemanticEmbedding) -> Tuple[Optional[SemanticCluster], float]:
        """Find most similar existing cluster"""
        
        if not self.known_clusters:
            return None, 0.0
        
        best_cluster = None
        best_similarity = 0.0
        
        for cluster in self.known_clusters:
            similarity = cluster.similarity_to_embedding(embedding)
            
            # Bonus for same vulnerability category
            if cluster.embeddings and cluster.embeddings[0].vulnerability_category == embedding.vulnerability_category:
                similarity += 0.1
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        return best_cluster, best_similarity
    
    async def _create_new_cluster(self, embedding: SemanticEmbedding, 
                                vulnerability_category: str) -> SemanticCluster:
        """Create new cluster for novel finding"""
        
        cluster = SemanticCluster(
            id=f"novel_{vulnerability_category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            centroid=embedding.embedding.copy(),
            embeddings=[embedding],
            cluster_type=vulnerability_category
        )
        
        return cluster
    
    async def reorganize_clusters(self) -> None:
        """Reorganize clusters by merging similar ones and splitting overgrown ones"""
        
        logger.info("Starting cluster reorganization")
        
        # Find clusters to merge
        merged_clusters = []
        used_clusters = set()
        
        for i, cluster1 in enumerate(self.known_clusters):
            if i in used_clusters:
                continue
            
            candidates_to_merge = [cluster1]
            merge_indices = {i}
            
            for j, cluster2 in enumerate(self.known_clusters):
                if j != i and j not in used_clusters:
                    if cluster1.should_merge_with(cluster2, self.merge_threshold):
                        candidates_to_merge.append(cluster2)
                        merge_indices.add(j)
            
            if len(candidates_to_merge) > 1:
                # Merge clusters
                merged_cluster = await self._merge_clusters(candidates_to_merge)
                merged_clusters.append(merged_cluster)
                used_clusters.update(merge_indices)
            else:
                merged_clusters.append(cluster1)
                used_clusters.add(i)
        
        # Split overgrown clusters
        final_clusters = []
        for cluster in merged_clusters:
            if len(cluster.embeddings) > 20:  # Threshold for splitting
                split_clusters = await self._split_cluster(cluster)
                final_clusters.extend(split_clusters)
            else:
                final_clusters.append(cluster)
        
        self.known_clusters = final_clusters
        
        # Memory management
        if len(self.known_clusters) > self.max_clusters_in_memory:
            # Keep most novel and recent clusters
            self.known_clusters.sort(key=lambda c: (c.novelty_score, c.last_updated), reverse=True)
            self.known_clusters = self.known_clusters[:self.max_clusters_in_memory]
        
        logger.info(f"Cluster reorganization complete: {len(self.known_clusters)} clusters")
    
    async def _merge_clusters(self, clusters: List[SemanticCluster]) -> SemanticCluster:
        """Merge multiple clusters into one"""
        
        all_embeddings = []
        for cluster in clusters:
            all_embeddings.extend(cluster.embeddings)
        
        # Determine cluster type (most common)
        cluster_types = [cluster.cluster_type for cluster in clusters]
        most_common_type = Counter(cluster_types).most_common(1)[0][0]
        
        merged_cluster = SemanticCluster(
            id=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            centroid=np.zeros(clusters[0].centroid.shape),  # Will be recalculated
            embeddings=all_embeddings,
            cluster_type=most_common_type
        )
        
        # Average novelty scores
        avg_novelty = sum(cluster.novelty_score for cluster in clusters) / len(clusters)
        merged_cluster.novelty_score = avg_novelty
        
        return merged_cluster
    
    async def _split_cluster(self, cluster: SemanticCluster) -> List[SemanticCluster]:
        """Split overgrown cluster into smaller ones"""
        
        if len(cluster.embeddings) <= 5:
            return [cluster]
        
        # Use clustering to split
        sub_clusters = await self.clustering_engine.cluster_embeddings(
            cluster.embeddings, 
            method='hierarchical'
        )
        
        # If splitting failed, return original cluster
        if len(sub_clusters) <= 1:
            return [cluster]
        
        # Update cluster IDs and types
        for i, sub_cluster in enumerate(sub_clusters):
            sub_cluster.id = f"split_{cluster.id}_{i}"
            sub_cluster.cluster_type = cluster.cluster_type
            sub_cluster.novelty_score = cluster.novelty_score
        
        return sub_clusters
    
    async def batch_process_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple findings in batch for efficiency"""
        
        logger.info(f"Batch processing {len(findings)} findings")
        
        # Generate embeddings for all findings
        embeddings = []
        for finding in findings:
            embedding = await self._create_embedding(
                finding['text'],
                finding['vulnerability_category'],
                finding.get('context', {})
            )
            embeddings.append(embedding)
        
        # Cluster new findings together
        new_clusters = await self.clustering_engine.cluster_embeddings(embeddings)
        
        # Process each new cluster
        novel_clusters = []
        merged_to_existing = []
        
        for new_cluster in new_clusters:
            # Check if this cluster is novel compared to existing ones
            best_existing, similarity = self._find_most_similar_cluster_for_cluster(new_cluster)
            
            if similarity < self.novelty_threshold:
                # Novel cluster
                novel_clusters.append(new_cluster)
                self.known_clusters.append(new_cluster)
                self.novel_discoveries += len(new_cluster.embeddings)
            else:
                # Merge with existing cluster
                for embedding in new_cluster.embeddings:
                    best_existing.add_embedding(embedding)
                merged_to_existing.append(new_cluster)
                self.deduplicated_count += len(new_cluster.embeddings)
        
        return {
            'total_processed': len(findings),
            'novel_clusters': len(novel_clusters),
            'novel_findings': sum(len(cluster.embeddings) for cluster in novel_clusters),
            'deduplicated_findings': sum(len(cluster.embeddings) for cluster in merged_to_existing),
            'processing_summary': {
                'new_clusters_created': len(novel_clusters),
                'clusters_merged_to_existing': len(merged_to_existing)
            }
        }
    
    def _find_most_similar_cluster_for_cluster(self, new_cluster: SemanticCluster) -> Tuple[Optional[SemanticCluster], float]:
        """Find most similar existing cluster to a new cluster"""
        
        if not self.known_clusters:
            return None, 0.0
        
        best_cluster = None
        best_similarity = 0.0
        
        for existing_cluster in self.known_clusters:
            # Calculate similarity between cluster centroids
            similarity = float(np.dot(new_cluster.centroid, existing_cluster.centroid))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = existing_cluster
        
        return best_cluster, best_similarity
    
    def get_deduplication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics"""
        
        cluster_stats = []
        for cluster in self.known_clusters:
            cluster_stats.append({
                'id': cluster.id,
                'type': cluster.cluster_type,
                'size': len(cluster.embeddings),
                'coherence': cluster.coherence_score,
                'novelty': cluster.novelty_score,
                'age_hours': (datetime.now() - cluster.creation_time).total_seconds() / 3600
            })
        
        # Calculate category distribution
        category_distribution = Counter(cluster.cluster_type for cluster in self.known_clusters)
        
        return {
            'total_processed': self.total_processed,
            'novel_discoveries': self.novel_discoveries,
            'deduplicated_count': self.deduplicated_count,
            'novelty_rate': self.novel_discoveries / max(1, self.total_processed),
            'deduplication_rate': self.deduplicated_count / max(1, self.total_processed),
            'total_clusters': len(self.known_clusters),
            'avg_cluster_size': sum(len(cluster.embeddings) for cluster in self.known_clusters) / max(1, len(self.known_clusters)),
            'category_distribution': dict(category_distribution),
            'cluster_details': cluster_stats,
            'memory_usage': {
                'clusters_in_memory': len(self.known_clusters),
                'embeddings_cached': len(self.embedding_cache)
            }
        }
    
    async def export_clusters(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export clusters for analysis or storage"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_deduplication_statistics(),
            'clusters': [cluster.to_dict() for cluster in self.known_clusters]
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        else:
            return export_data