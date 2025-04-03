"""
This script checks the import of DQNAgent, RainbowDQNAgent, ActorCriticAgent, and CategoricalQNetwork.
It ensures that all necessary modules are correctly imported and available for use.
"""

import sys
sys.path.insert(0, '.')

try:
    print("Trying to import DQNAgent...")
    from qtrust.agents.dqn.agent import DQNAgent
    print("✓ DQNAgent import successful!")
except Exception as e:
    print(f"✗ Error importing DQNAgent: {e}")

try:
    print("\nTrying to import RainbowDQNAgent...")
    from qtrust.agents.dqn.rainbow_agent import RainbowDQNAgent
    print("✓ RainbowDQNAgent import successful!")
except Exception as e:
    print(f"✗ Error importing RainbowDQNAgent: {e}")

try:
    print("\nTrying to import ActorCriticAgent...")
    from qtrust.agents.dqn.actor_critic_agent import ActorCriticAgent
    print("✓ ActorCriticAgent import successful!")
except Exception as e:
    print(f"✗ Error importing ActorCriticAgent: {e}")

print("\nChecking dependencies and imports for RainbowDQNAgent:")
try:
    print("Trying to import numpy, torch...")
    import numpy as np
    import torch
    print("✓ numpy and torch import successful!")
except Exception as e:
    print(f"✗ Error importing numpy, torch: {e}")

try:
    print("\nTrying to import CategoricalQNetwork...")
    from qtrust.agents.dqn.networks import CategoricalQNetwork
    print("✓ CategoricalQNetwork import successful!")
except Exception as e:
    print(f"✗ Error importing CategoricalQNetwork: {e}")

try:
    print("\nTrying to import NStepPrioritizedReplayBuffer...")
    from qtrust.agents.dqn.replay_buffer import NStepPrioritizedReplayBuffer
    print("✓ NStepPrioritizedReplayBuffer import successful!")
except Exception as e:
    print(f"✗ Error importing NStepPrioritizedReplayBuffer: {e}")

print("\nCheck complete!") 