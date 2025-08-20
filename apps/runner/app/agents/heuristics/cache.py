# ABOUTME: High-performance evaluation caching system for cost optimization
# ABOUTME: SHA-256 hash-based caching with LRU eviction and monitoring

import time
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock

from apps.runner.app.util.schemas import Attempt, Evaluation


@dataclass
class CacheEntry:
    """Entry in the evaluation cache"""
    evaluation: Evaluation
    created_at: float  # Timestamp when cached
    last_accessed: float  # Last access time for LRU
    access_count: int  # Number of times accessed
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired based on TTL"""
        return time.time() - self.created_at > ttl_seconds


class EvaluationCache:
    """
    High-performance caching system for evaluation results.
    
    Features:
    - SHA-256 hash-based keys for collision resistance
    - LRU eviction when at capacity
    - Configurable TTL (time-to-live)
    - Async-safe operations with thread safety
    - Performance monitoring and hit rate tracking
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        """
        Initialize evaluation cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        
        # Thread-safe cache storage using OrderedDict for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()  # For thread safety
        
        # Performance monitoring
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.cleanup_count = 0
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_cleanup()
    
    def _generate_cache_key(self, attempt: Attempt, category: Optional[str] = None) -> str:
        """
        Generate SHA-256 hash key for caching.
        Key includes: category + prompt_hash + response_hash for uniqueness.
        
        Args:
            attempt: Attempt to generate key for
            category: Safety category (if available)
            
        Returns:
            32-character SHA-256 hash for cache key
        """
        # Include category in key for category-specific caching
        category_str = category or attempt.metadata.get('category', 'unknown') if attempt.metadata else 'unknown'
        
        # Create composite content for hashing
        content_parts = [
            category_str,
            attempt.prompt or "",
            attempt.response or ""
        ]
        
        content = "|".join(content_parts)
        hash_object = hashlib.sha256(content.encode('utf-8'))
        
        # Return first 32 characters for reasonable key length
        return hash_object.hexdigest()[:32]
    
    async def get(self, attempt: Attempt, category: Optional[str] = None) -> Optional[Evaluation]:
        """
        Retrieve evaluation from cache if available and not expired.
        
        Args:
            attempt: Attempt to look up
            category: Safety category for key generation
            
        Returns:
            Cached Evaluation or None if not found/expired
        """
        cache_key = self._generate_cache_key(attempt, category)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self.miss_count += 1
                return None
            
            # Check if entry is expired
            if entry.is_expired(self.ttl_seconds):
                del self._cache[cache_key]
                self.miss_count += 1
                self.cleanup_count += 1
                return None
            
            # Update access tracking for LRU
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end for LRU ordering
            self._cache.move_to_end(cache_key)
            
            self.hit_count += 1
            return entry.evaluation
    
    async def put(self, attempt: Attempt, evaluation: Evaluation, category: Optional[str] = None):
        """
        Store evaluation result in cache with LRU eviction if needed.
        
        Args:
            attempt: Attempt that was evaluated
            evaluation: Evaluation result to cache
            category: Safety category for key generation
        """
        cache_key = self._generate_cache_key(attempt, category)
        current_time = time.time()
        
        with self._lock:
            # Create new cache entry
            entry = CacheEntry(
                evaluation=evaluation,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0
            )
            
            # Check if we need to evict entries (LRU)
            while len(self._cache) >= self.max_size:
                # Remove least recently used item (first in OrderedDict)
                oldest_key, _ = self._cache.popitem(last=False)
                self.eviction_count += 1
            
            # Add new entry (will be at end of OrderedDict)
            self._cache[cache_key] = entry
    
    def invalidate(self, attempt: Attempt, category: Optional[str] = None):
        """
        Remove specific entry from cache.
        
        Args:
            attempt: Attempt to invalidate
            category: Safety category for key generation
        """
        cache_key = self._generate_cache_key(attempt, category)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self.hit_count = 0
            self.miss_count = 0
            self.eviction_count = 0
            self.cleanup_count = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Current number of cached entries"""
        return len(self._cache)
    
    @property
    def capacity_used(self) -> float:
        """Percentage of cache capacity used"""
        return (len(self._cache) / self.max_size * 100) if self.max_size > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics for monitoring.
        
        Returns:
            Dictionary with cache performance metrics
        """
        total_requests = self.hit_count + self.miss_count
        
        return {
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "capacity_used_percent": self.capacity_used,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": self.hit_rate,
            "eviction_count": self.eviction_count,
            "cleanup_count": self.cleanup_count,
            "total_requests": total_requests,
            "ttl_hours": self.ttl_seconds / 3600,
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """
        Calculate cache efficiency score (0-100).
        Factors in hit rate, eviction rate, and capacity utilization.
        """
        if self.hit_count + self.miss_count == 0:
            return 0.0
        
        hit_rate = self.hit_rate / 100.0  # Normalize to 0-1
        
        # Factor in eviction rate (lower is better)
        total_entries = self.hit_count + self.miss_count + self.eviction_count
        eviction_rate = self.eviction_count / total_entries if total_entries > 0 else 0
        eviction_penalty = min(eviction_rate * 2, 0.5)  # Cap penalty at 50%
        
        # Factor in capacity utilization (sweet spot around 70-90%)
        capacity_factor = min(self.capacity_used / 100.0, 1.0)
        if capacity_factor > 0.9:
            capacity_factor = capacity_factor * 0.9  # Slight penalty for over-utilization
        
        efficiency = hit_rate * (1 - eviction_penalty) * capacity_factor
        return efficiency * 100.0  # Return as percentage
    
    async def cleanup_expired(self):
        """
        Manual cleanup of expired entries.
        Called periodically by background task.
        """
        expired_keys = []
        current_time = time.time()
        
        with self._lock:
            for key, entry in list(self._cache.items()):
                if entry.is_expired(self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self.cleanup_count += 1
    
    def _start_background_cleanup(self):
        """Start background task for periodic cleanup"""
        async def cleanup_task():
            while True:
                await asyncio.sleep(600)  # Cleanup every 10 minutes
                await self.cleanup_expired()
        
        # Only start if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._cleanup_task = loop.create_task(cleanup_task())
        except RuntimeError:
            # No event loop running - cleanup will be manual only
            pass
    
    def __del__(self):
        """Cleanup background task on destruction"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class CacheMonitor:
    """
    Monitor cache performance and provide optimization recommendations.
    """
    
    def __init__(self, cache: EvaluationCache):
        self.cache = cache
        self.last_stats_snapshot = None
        self.stats_history = []
        self.max_history = 100  # Keep last 100 snapshots
    
    def take_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current cache statistics"""
        current_stats = self.cache.get_statistics()
        current_stats['timestamp'] = time.time()
        
        # Add to history
        self.stats_history.append(current_stats)
        if len(self.stats_history) > self.max_history:
            self.stats_history.pop(0)
        
        self.last_stats_snapshot = current_stats
        return current_stats
    
    def get_recommendations(self) -> List[str]:
        """
        Get optimization recommendations based on cache performance.
        
        Returns:
            List of recommendation strings
        """
        if not self.last_stats_snapshot:
            self.take_snapshot()
        
        stats = self.last_stats_snapshot
        recommendations = []
        
        # Hit rate recommendations
        hit_rate = stats['hit_rate_percent']
        if hit_rate < 30:
            recommendations.append("Cache hit rate is low (<30%). Consider increasing TTL or cache size.")
        elif hit_rate > 85:
            recommendations.append("Excellent cache hit rate (>85%). Current configuration is optimal.")
        
        # Capacity recommendations
        capacity_used = stats['capacity_used_percent']
        if capacity_used > 95:
            recommendations.append("Cache is near capacity (>95%). Consider increasing max_size to reduce evictions.")
        elif capacity_used < 20:
            recommendations.append("Cache is under-utilized (<20%). Consider reducing max_size to save memory.")
        
        # Eviction rate recommendations
        if stats['total_requests'] > 100:  # Only after sufficient activity
            eviction_rate = stats['eviction_count'] / stats['total_requests']
            if eviction_rate > 0.1:
                recommendations.append("High eviction rate (>10%). Consider increasing cache size or reducing TTL.")
        
        return recommendations