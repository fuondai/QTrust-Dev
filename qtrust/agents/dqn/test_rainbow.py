import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

from .agent import DQNAgent
from .rainbow_agent import RainbowDQNAgent
from .actor_critic_agent import ActorCriticAgent

def test_rainbow_dqn(
    env: gym.Env,
    num_episodes: int = 5,
    max_steps: int = 100,
    render: bool = False,
    seed: int = 42
) -> Tuple[RainbowDQNAgent, List[float], List[float]]:
    """
    Test Rainbow DQN agent on a given environment.
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        buffer_size=200,
        batch_size=4,
        n_step=2,
        n_atoms=11,
        v_min=-10.0,
        v_max=10.0,
        hidden_layers=[64, 32],
        learning_rate=1e-3,
        update_every=1,
        warm_up_steps=4
    )
    
    # Training loop
    scores = []
    avg_scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        score = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update agent
            agent.step(state, action, reward, next_state, done)
            
            score += reward
            state = next_state
            
            if done or truncated:
                break
        
        scores.append(score)
        avg_scores.append(np.mean(scores))
        
    return agent, scores, avg_scores

def test_actor_critic(
    env: gym.Env,
    num_episodes: int = 5,
    max_steps: int = 100,
    render: bool = False,
    seed: int = 42
) -> Tuple[ActorCriticAgent, List[float], List[float]]:
    """
    Test Actor-Critic agent on a given environment.
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    agent = ActorCriticAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_layers=[64, 32],
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_size=200,
        batch_size=4,
        update_every=1,
        warm_up_steps=4
    )
    
    # Training loop
    scores = []
    avg_scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        score = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update agent
            agent.step(state, action, reward, next_state, done)
            
            score += reward
            state = next_state
            
            if done or truncated:
                break
        
        scores.append(score)
        avg_scores.append(np.mean(scores))
        
    return agent, scores, avg_scores

def compare_methods(
    env: gym.Env,
    num_episodes: int = 5,
    max_steps: int = 100,
    seed: int = 42
) -> Tuple[List[Any], List[List[float]], List[List[float]]]:
    """
    Compare different RL methods on a given environment.
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agents
    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=64,
        buffer_size=200,
        batch_size=4,
        learning_rate=1e-3,
        update_every=1
    )
    
    rainbow_agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        buffer_size=200,
        batch_size=4,
        n_step=2,
        n_atoms=11,
        v_min=-10.0,
        v_max=10.0,
        hidden_layers=[64, 32],
        learning_rate=1e-3,
        update_every=1,
        warm_up_steps=4
    )
    
    actor_critic_agent = ActorCriticAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_layers=[64, 32],
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_size=200,
        batch_size=4,
        update_every=1,
        warm_up_steps=4
    )
    
    agents = [dqn_agent, rainbow_agent, actor_critic_agent]
    all_scores = []
    all_avg_scores = []
    
    # Train each agent
    for agent in agents:
        scores = []
        avg_scores = []
        
        for episode in range(num_episodes):
            state, _ = env.reset(seed=seed)
            score = 0
            
            for step in range(max_steps):
                # Select action
                action = agent.act(state)
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Update agent
                agent.step(state, action, reward, next_state, done)
                
                score += reward
                state = next_state
                
                if done or truncated:
                    break
            
            scores.append(score)
            avg_scores.append(np.mean(scores))
        
        all_scores.append(scores)
        all_avg_scores.append(avg_scores)
    
    # Plot results
    agent_names = ['DQN', 'Rainbow DQN', 'Actor-Critic']
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_scores[i] for i in range(len(agents))], labels=agent_names)
    plt.title('Performance Comparison')
    plt.ylabel('Score')
    plt.savefig('agents_comparison.png')
    plt.close()
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    for i, name in enumerate(agent_names):
        scores = all_scores[i]
        print(f"{name}:")
        print(f"  Average score: {np.mean(scores):.2f}")
        print(f"  Highest score: {np.max(scores):.2f}")
        print(f"  Lowest score: {np.min(scores):.2f}")
        print(f"  Standard deviation: {np.std(scores):.2f}")
    
    return agents, all_scores, all_avg_scores 