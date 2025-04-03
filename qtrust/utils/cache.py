"""
Caching utilities for QTrust application.

This module provides classes and utility functions for implementing caching, minimizing
redundant computations and improving overall system performance.

It includes implementations of LRU and TTL caches, as well as decorators for easily
applying caching to functions, with special support for PyTorch tensors.
"""

import time
import functools
import hashlib
import pickle
import logging
from typing import Dict, Any, Callable, Tuple, List, Union, Optional, TypeVar, Generic

import numpy as np
import torch

# Setup logger
logger = logging.getLogger("qtrust.cache")

# Type variables for generics
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class LRUCache(Generic[K, V]):
    """
    Implementation of LRU (Least Recently Used) cache.
    
    This class manages cache using the LRU algorithm, removing the least recently used
    items when the cache size reaches its limit.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize LRUCache.
        
        Args:
            capacity: Maximum number of items in the cache
        """
        self.capacity = capacity
        self.cache: Dict[K, V] = {}
        self.usage_order: List[K] = []
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get a value from the cache. Update usage order.
        
        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Cached value or default value
        """
        if key not in self.cache:
            return default
        
        # Update usage order
        self.usage_order.remove(key)
        self.usage_order.append(key)
        
        return self.cache[key]
    
    def put(self, key: K, value: V) -> None:
        """
        Add or update a value in the cache.
        
        Args:
            key: Key
            value: Value
        """
        if key in self.cache:
            # Update usage order
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            oldest = self.usage_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.usage_order.append(key)
    
    def __contains__(self, key: K) -> bool:
        """Check if key is in the cache."""
        return key in self.cache
    
    def __len__(self) -> int:
        """Number of items in the cache."""
        return len(self.cache)
    
    def clear(self) -> None:
        """Remove all items from the cache."""
        self.cache.clear()
        self.usage_order.clear()

class TTLCache(Generic[K, V]):
    """
    Implementation of TTL (Time-To-Live) cache.
    
    This class manages cache with a time-to-live for each item, automatically
    removing expired items.
    """
    
    def __init__(self, ttl: float = 300.0, capacity: int = 1000):
        """
        Initialize TTLCache.
        
        Args:
            ttl: Time-to-live (seconds) for each item
            capacity: Maximum number of items in the cache
        """
        self.ttl = ttl
        self.capacity = capacity
        self.cache: Dict[K, Tuple[V, float]] = {}  # (value, expiration_time)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get a value from the cache if it hasn't expired.
        
        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist or has expired
            
        Returns:
            Cached value or default value
        """
        if key not in self.cache:
            return default
        
        value, expiration_time = self.cache[key]
        
        # Check if the item has expired
        if time.time() > expiration_time:
            del self.cache[key]
            return default
        
        return value
    
    def put(self, key: K, value: V) -> None:
        """
        Add or update a value in the cache with a new expiration time.
        
        Args:
            key: Key
            value: Value
        """
        # Clean expired entries
        self._clean_expired()
        
        # Check if cache is full
        if len(self.cache) >= self.capacity and key not in self.cache:
            # Remove oldest expired entry
            self._remove_oldest()
        
        # Calculate new expiration time
        expiration_time = time.time() + self.ttl
        
        # Add or update item in cache
        self.cache[key] = (value, expiration_time)
    
    def _clean_expired(self) -> None:
        """Remove all expired items from the cache."""
        current_time = time.time()
        expired_keys = [k for k, (_, exp_time) in self.cache.items() if current_time > exp_time]
        for key in expired_keys:
            del self.cache[key]
    
    def _remove_oldest(self) -> None:
        """Remove the item with the earliest expiration time."""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        del self.cache[oldest_key]
    
    def __contains__(self, key: K) -> bool:
        """Check if key is in the cache and hasn't expired."""
        if key not in self.cache:
            return False
        
        _, expiration_time = self.cache[key]
        if time.time() > expiration_time:
            del self.cache[key]
            return False
        
        return True
    
    def __len__(self) -> int:
        """Number of items in the cache (including expired items)."""
        return len(self.cache)
    
    def clear(self) -> None:
        """Remove all items from the cache."""
        self.cache.clear()

def compute_hash(obj: Any) -> str:
    """
    Compute hash code for an object.
    
    Supports NumPy and PyTorch objects.
    
    Args:
        obj: Object to hash
        
    Returns:
        str: Hash string
    """
    # Special case for tensors
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    
    # Special case for NumPy arrays
    if isinstance(obj, np.ndarray):
        obj_bytes = obj.tobytes()
    else:
        try:
            # Try to serialize the object
            obj_bytes = pickle.dumps(obj)
        except (pickle.PickleError, TypeError):
            # Fallback to handle objects that can't be pickled
            obj_bytes = str(obj).encode('utf-8')
    
    # Compute hash
    return hashlib.md5(obj_bytes).hexdigest()

def lru_cache(maxsize: int = 128):
    """
    Decorator to cache function results with LRU cache.
    
    Args:
        maxsize: Maximum number of results to cache
        
    Returns:
        Decorator
    """
    cache = LRUCache(capacity=maxsize)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from parameters
            key_parts = [compute_hash(arg) for arg in args]
            key_parts.extend(f"{k}:{compute_hash(v)}" for k, v in sorted(kwargs.items()))
            key = f"{func.__name__}:{'-'.join(key_parts)}"
            
            # Check cache
            if key in cache:
                return cache.get(key)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(key, result)
            
            return result
        
        # Add reference to cache to be able to clear it if needed
        wrapper.cache = cache
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": cache.capacity}
        wrapper.cache_clear = cache.clear
        
        return wrapper
    
    return decorator

def ttl_cache(ttl: float = 300.0, maxsize: int = 128):
    """
    Decorator to cache function results with TTL cache.
    
    Args:
        ttl: Time-to-live (seconds) for each item
        maxsize: Maximum number of results to cache
        
    Returns:
        Decorator
    """
    cache = TTLCache(ttl=ttl, capacity=maxsize)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from parameters
            key_parts = [compute_hash(arg) for arg in args]
            key_parts.extend(f"{k}:{compute_hash(v)}" for k, v in sorted(kwargs.items()))
            key = f"{func.__name__}:{'-'.join(key_parts)}"
            
            # Check cache
            if key in cache:
                return cache.get(key)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(key, result)
            
            return result
        
        # Add reference to cache to be able to clear it if needed
        wrapper.cache = cache
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": cache.capacity, "ttl": cache.ttl}
        wrapper.cache_clear = cache.clear
        
        return wrapper
    
    return decorator

def tensor_cache(func: Callable):
    """
    Special decorator to cache PyTorch Tensor computation results.
    
    Args:
        func: Function to cache
        
    Returns:
        Wrapped function
    """
    # Use LRU cache with default size
    cache = LRUCache(capacity=128)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert all Tensors to NumPy for hashing
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Only use shape and a portion of data to create hash
                shape = arg.shape
                dtype = str(arg.dtype)
                sample = arg.detach().flatten()[:min(100, arg.numel())].cpu().numpy()
                processed_args.append((shape, dtype, sample))
            else:
                processed_args.append(arg)
        
        processed_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # Same as args
                shape = v.shape
                dtype = str(v.dtype)
                sample = v.detach().flatten()[:min(100, v.numel())].cpu().numpy()
                processed_kwargs[k] = (shape, dtype, sample)
            else:
                processed_kwargs[k] = v
        
        # Create key from processed parameters
        key_parts = [compute_hash(arg) for arg in processed_args]
        key_parts.extend(f"{k}:{compute_hash(v)}" for k, v in sorted(processed_kwargs.items()))
        key = f"{func.__name__}:{'-'.join(key_parts)}"
        
        # Check cache
        if key in cache:
            return cache.get(key)
        
        # Compute result
        result = func(*args, **kwargs)
        
        # Only cache if result is tensor and doesn't require gradient
        if isinstance(result, torch.Tensor) and not result.requires_grad:
            cache.put(key, result.clone().detach())
        elif isinstance(result, (list, tuple)) and all(isinstance(r, torch.Tensor) and not r.requires_grad for r in result):
            cloned_result = tuple(r.clone().detach() for r in result)
            cache.put(key, cloned_result)
        else:
            # Don't cache results with gradients or non-tensor results
            return result
        
        return result
    
    # Add reference to cache to be able to clear it if needed
    wrapper.cache = cache
    wrapper.cache_info = lambda: {"size": len(cache), "maxsize": cache.capacity}
    wrapper.cache_clear = cache.clear
    
    return wrapper 