"""
Simple DQN Cache Test
This script tests the caching functionality of the DQNAgent.
"""

import time
import numpy as np
from qtrust.agents.dqn.agent import DQNAgent

def main():
    """Test DQNAgent caching functionality."""
    # Create a simple agent
    state_size = 4
    action_size = 2
    
    print("Creating DQNAgent...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_layers=[64, 64],
        buffer_size=1000,
        batch_size=64,
        prioritized_replay=False
    )
    
    print("Testing action caching...")
    
    # Create a fixed state for testing
    state = np.array([0.1, 0.2, 0.3, 0.4])
    
    # First call - should be a cache miss
    start_time = time.time()
    action1 = agent.act(state, eps=0.0)  # Force greedy action
    first_call_time = time.time() - start_time
    print(f"First call: action={action1}, time={first_call_time*1000:.2f}ms")
    
    # Second call with same state - should be a cache hit
    start_time = time.time()
    action2 = agent.act(state, eps=0.0)  # Force greedy action  
    second_call_time = time.time() - start_time
    print(f"Second call: action={action2}, time={second_call_time*1000:.2f}ms")
    
    # Compare performance
    if hasattr(agent, 'cache_hits') and hasattr(agent, 'cache_misses'):
        print(f"Cache hits: {agent.cache_hits}")
        print(f"Cache misses: {agent.cache_misses}")
        if agent.cache_hits + agent.cache_misses > 0:
            cache_hit_ratio = agent.cache_hits / (agent.cache_hits + agent.cache_misses)
            print(f"Cache hit ratio: {cache_hit_ratio*100:.1f}%")
    else:
        print("Agent does not have cache hit/miss tracking")
    
    # Check if second call was faster
    speedup = first_call_time / second_call_time if second_call_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    
    # Clear cache and test again
    if hasattr(agent, 'clear_cache'):
        print("\nClearing cache...")
        agent.clear_cache()
        
        # After clearing, this should be a cache miss again
        start_time = time.time()
        action3 = agent.act(state, eps=0.0)
        third_call_time = time.time() - start_time
        print(f"After clear: action={action3}, time={third_call_time*1000:.2f}ms")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 