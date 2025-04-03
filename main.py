#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust - Optimized Blockchain Sharding with Deep Reinforcement Learning

This file is the main entry point for running QTrust simulations. It orchestrates the entire
process including environment setup, agent initialization, training, and evaluation of the
blockchain sharding optimization system using Deep Reinforcement Learning.
"""

import sys
import os
import locale
import copy

# Force UTF-8 encoding for console output
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Try to set locale to UTF-8
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        pass

# Add the current directory to Python path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
import random
from pathlib import Path
from gym import spaces
import json
from datetime import datetime

from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.agents.dqn.agent import DQNAgent
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.trust.htdcm import HTDCM
from qtrust.federated.federated_learning import FederatedLearning, FederatedModel, FederatedClient
from qtrust.utils.metrics import (
    calculate_throughput, 
    calculate_latency_metrics,
    calculate_energy_efficiency,
    calculate_security_metrics,
    calculate_cross_shard_transaction_ratio,
    plot_performance_metrics,
    plot_comparison_charts
)
from qtrust.utils.data_generation import (
    generate_network_topology,
    assign_nodes_to_shards,
    generate_transactions
)
# Bỏ comment các dòng import quan trọng
from qtrust.federated.manager import FederatedLearningManager
from qtrust.consensus.adaptive_pos import AdaptivePoSManager

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Q-TRUST: Optimized Blockchain Sharding with DRL')
    
    # Blockchain environment parameters
    parser.add_argument('--num-shards', type=int, default=4, help='Number of shards')
    parser.add_argument('--nodes-per-shard', type=int, default=6, help='Number of nodes in each shard')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps in each episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    
    # Federated Learning parameters
    parser.add_argument('--enable-federated', action='store_true', help='Enable Federated Learning')
    
    # DQN parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=128, help='Number of neurons in hidden layer')
    parser.add_argument('--memory-size', type=int, default=10000, help='Replay memory size')
    parser.add_argument('--target-update', type=int, default=10, help='Episodes before target network update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon for ε-greedy')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon for ε-greedy')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Evaluation mode parameters
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--attack-scenario', type=str, default='none', 
                      choices=['none', 'ddos', '51_percent', 'sybil', 'eclipse'], 
                      help='Attack scenario for simulation')
    
    # Consensus parameters
    parser.add_argument('--enable-bls', action='store_true', help='Enable BLS signature aggregation')
    parser.add_argument('--enable-adaptive-pos', action='store_true', help='Enable Adaptive PoS with validator rotation')
    parser.add_argument('--enable-lightweight-crypto', action='store_true', help='Enable lightweight cryptography for energy optimization')
    parser.add_argument('--energy-optimization-level', type=str, choices=['low', 'balanced', 'aggressive'], default='balanced', 
                       help='Energy optimization level (low, balanced, aggressive)')
    parser.add_argument('--pos-rotation-period', type=int, default=50, help='Number of rounds before considering validator rotation')
    parser.add_argument('--active-validator-ratio', type=float, default=0.7, help='Ratio of active validators (0.0-1.0)')
    
    # Storage
    parser.add_argument('--save-dir', type=str, default='models/simulation', help='Directory to save results')
    parser.add_argument('--load-model', type=str, help='Path to model for continued training')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluation frequency (episodes)')
    parser.add_argument('--log-interval', type=int, default=5, help='Logging frequency (steps)')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    return args

def setup_simulation(args):
    """
    Set up the simulation environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple containing environment, router, HTDCM, and adaptive consensus manager
    """
    print("Initializing simulation environment...")
    
    # Initialize blockchain environment
    env = BlockchainEnvironment(
        num_shards=args.num_shards,
        num_nodes_per_shard=args.nodes_per_shard,
        max_steps=args.max_steps
    )
    
    # Set up router
    router = MADRAPIDRouter(
        network=env.network,
        shards=env.shards,
        congestion_weight=0.4,
        latency_weight=0.3,
        energy_weight=0.2,
        trust_weight=0.1
    )
    
    # Initialize HTDCM
    htdcm = HTDCM(num_nodes=args.num_shards * args.nodes_per_shard)
    
    # Set up adaptive consensus manager
    ac_manager = AdaptiveConsensus(
        transaction_threshold_low=10.0,
        transaction_threshold_high=50.0,
        congestion_threshold=0.7,
        min_trust_threshold=0.3,
        enable_bls=args.enable_bls,
        num_validators_per_shard=args.nodes_per_shard,
        enable_adaptive_pos=args.enable_adaptive_pos,
        enable_lightweight_crypto=args.enable_lightweight_crypto,
        active_validator_ratio=args.active_validator_ratio,
        rotation_period=args.pos_rotation_period
    )
    
    return env, router, htdcm, ac_manager

def setup_dqn_agent(env, args):
    """
    Set up the DQN Agent.
    
    Args:
        env: Blockchain environment
        args: Command line arguments
        
    Returns:
        Wrapped DQN agent
    """
    print("Initializing DQN Agent...")
    
    # Print environment observation space size for debugging
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Number of shards: {env.num_shards}")
    
    # Get state size from environment
    # Need to ensure correct input dimension
    state = env.reset()
    state_size = len(state)
    print(f"Actual state size: {state_size}")
    
    # Create wrapper for agent to convert from single action to MultiDiscrete action
    class DQNAgentWrapper:
        def __init__(self, agent, num_shards, num_consensus_protocols=3):
            self.agent = agent
            self.num_shards = num_shards
            self.num_consensus_protocols = num_consensus_protocols
            
        def act(self, state, eps=None):
            # Get action from base agent
            action_idx = self.agent.act(state, eps)
            
            # Convert action_idx to MultiDiscrete action [shard_idx, consensus_idx]
            shard_idx = action_idx % self.num_shards
            consensus_idx = (action_idx // self.num_shards) % self.num_consensus_protocols
            
            return np.array([shard_idx, consensus_idx], dtype=np.int32)
            
        def step(self, state, action, reward, next_state, done):
            # Convert MultiDiscrete action to single action
            if isinstance(action, np.ndarray) and len(action) >= 2:
                action_idx = action[0] + action[1] * self.num_shards
            else:
                # Handle case where action is an integer
                action_idx = action
            
            # Call step method of base agent
            self.agent.step(state, action_idx, reward, next_state, done)
            
        def save(self, path):
            return self.agent.save(path)
            
        def load(self, path):
            return self.agent.load(path)
            
        # Forward property
        @property
        def epsilon(self):
            return self.agent.epsilon
        
        # Add any additional properties if needed
        @property
        def device(self):
            return self.agent.device
            
    # Calculate total possible actions
    total_actions = env.num_shards * 3  # num_shards * num_consensus_protocols
    
    # Initialize base agent
    base_agent = DQNAgent(
        state_size=state_size,
        action_size=total_actions,  # Total possible actions
        seed=SEED,
        buffer_size=args.memory_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.lr,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.epsilon_end,
        hidden_layers=[args.hidden_size, args.hidden_size//2],
        device=args.device,
        prioritized_replay=True,
        dueling=True,
        update_every=5
    )
    
    # Create wrapper
    agent = DQNAgentWrapper(base_agent, env.num_shards)
    
    # If model path is provided, load trained model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading trained model from: {args.model_path}")
        agent.load(args.model_path)
        print("Successfully loaded DQN model")
        
        # If in evaluation mode, set epsilon to 0 to avoid random exploration
        if args.eval:
            base_agent.epsilon = 0.0
            print("Set epsilon = 0 for evaluation mode (no random exploration)")
    else:
        if args.model_path:
            print(f"Warning: Model not found at {args.model_path}, using new model")
        else:
            print("Initializing new DQN model")
    
    return agent

def setup_federated_learning(env, args, htdcm):
    """
    Set up the Federated Learning system.
    
    Note: The Federated Learning system is currently under development. While the components
    are initialized here, the actual FL training process is not fully integrated into the
    main training loop yet. To use FL, you need to:
    1. Enable it with --enable-federated flag
    2. Add code in the main training loop to periodically call fl_system.train_round()
       with the collected shard_experiences data
    
    Args:
        env: Blockchain environment
        args: Command line arguments
        htdcm: Hierarchical Trust and Distributed Consensus Management system
        
    Returns:
        FederatedLearning system or None if federated learning is disabled
    """
    if not args.enable_federated:
        return None
    
    print("Initializing Federated Learning system...")
    
    # Initialize global model
    # Get actual state size instead of from observation_space
    state = env.reset()
    input_size = len(state)
    hidden_size = args.hidden_size
    
    # Handle case where action_space is MultiDiscrete
    if hasattr(env.action_space, 'n'):
        output_size = env.action_space.n
    else:
        # For MultiDiscrete, get total number of available actions
        output_size = env.action_space.nvec.sum()
    
    global_model = FederatedModel(input_size, hidden_size, output_size)
    
    # Initialize Federated Learning system
    fl_system = FederatedLearning(
        global_model=global_model,
        aggregation_method='fedtrust',
        min_clients_per_round=3,
        device=args.device
    )
    
    # Create a client for each shard
    for shard_id in range(env.num_shards):
        # Get average trust score for the shard - use default if not available
        try:
            shard_trust = htdcm.shard_trust_scores.get(shard_id, 0.7)
        except (AttributeError, IndexError):
            # Nếu không có shard_trust_scores, sử dụng giá trị mặc định
            shard_trust = 0.7
        
        # Create new client
        client = FederatedClient(
            client_id=shard_id,
            model=copy.deepcopy(global_model),  # Mỗi client cần một bản sao của mô hình
            learning_rate=args.lr,
            local_epochs=5,
            batch_size=32,
            trust_score=shard_trust,
            device=args.device
        )
        
        # Thêm client vào hệ thống FL
        fl_system.add_client(client)
    
    print(f"Federated Learning đã được khởi tạo với {env.num_shards} clients")
    return fl_system

def train_qtrust(env, agent, router, consensus, htdcm, fl_system, args):
    """
    Train the QTrust system.
    
    Args:
        env: Blockchain environment
        agent: DQN agent
        router: Transaction router
        consensus: Adaptive consensus manager
        htdcm: Hierarchical Trust system
        fl_system: Federated Learning system
        args: Command line arguments
        
    Returns:
        Dictionary of training metrics
    """
    print("Starting QTrust system training...")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Track performance
    episode_rewards = []
    avg_rewards = []
    transaction_throughputs = []
    latencies = []
    malicious_detections = []
    
    # Train for multiple episodes
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Store data for federated learning
        shard_experiences = [[] for _ in range(env.num_shards)]
        
        # Run one episode
        while not done and steps < args.max_steps:
            # Select action from DQN Agent
            action = agent.act(state)
            
            # Simulate activities in the blockchain network
            
            # 1. Route transactions
            transaction_routes = router.find_optimal_paths_for_transactions(env.transaction_pool)
            
            # 2. Execute transactions with adaptive consensus protocols
            for tx in env.transaction_pool:
                # Select consensus protocol based on transaction value and network conditions
                trust_scores_dict = htdcm.get_node_trust_scores()
                
                # Ước tính mức độ tắc nghẽn của toàn mạng từ shard_congestion
                network_congestion = np.mean(list(env.shard_congestion.values())) if isinstance(env.shard_congestion, dict) else 0.5
                
                # Lấy shard_id từ transaction
                shard_id = tx.get('source_shard', 0)
                
                try:
                    protocol = consensus.select_protocol(
                        transaction_value=tx['value'],
                        congestion=network_congestion,
                        trust_scores=trust_scores_dict
                    )
                    
                    # Execute consensus protocol - lấy 4 giá trị trả về
                    result, protocol_name, latency, energy = consensus.execute_consensus(
                        transaction_value=tx['value'],
                        congestion=network_congestion,
                        trust_scores=trust_scores_dict,
                        network_stability=0.5,  # Default value for stability
                        cross_shard=(tx.get('type') == 'cross_shard'),
                        shard_id=shard_id
                    )
                    # Lưu protocol name vào transaction
                    tx['protocol'] = protocol_name
                except Exception as e:
                    print(f"Error executing consensus for transaction: {str(e)}")
                    # Fallback values
                    result, latency, energy = True, 20.0, 50.0
                    tx['protocol'] = "PBFT"  # Default protocol
                
                # Record results in the trust system
                for node_id in tx.get('validator_nodes', []):
                    htdcm.update_node_trust(
                        node_id=node_id,
                        tx_success=result,
                        response_time=latency,
                        is_validator=True
                    )
            
            # 3. Execute step in the environment with the selected action
            next_state, reward, done, info = env.step(action)
            
            # 4. Update Agent
            agent.step(state, action, reward, next_state, done)
            
            # Store experience for each shard
            for shard_id in range(env.num_shards):
                # Get transactions related to this shard
                shard_txs = [tx for tx in env.transaction_pool 
                            if tx.get('shard_id') == shard_id or tx.get('destination_shard') == shard_id]
                
                if shard_txs:
                    # Store experience (state, action, reward, next_state)
                    shard_experience = (state, action, reward, next_state, done)
                    shard_experiences[shard_id].append(shard_experience)
            
            # Thực hiện Federated Learning định kỳ nếu được bật
            if fl_system is not None and steps % 10 == 0:
                try:
                    print(f"Thực hiện Federated Learning tại bước {steps}...")
                    
                    # Thu thập dữ liệu từ các shard cho huấn luyện
                    active_shards = []
                    for shard_id, experiences in enumerate(shard_experiences):
                        if experiences:  # Chỉ xét các shard có dữ liệu
                            active_shards.append(shard_id)
                            
                            # Lấy client tương ứng với shard_id
                            client = fl_system.clients.get(shard_id)
                            if client:
                                # Chuyển đổi kinh nghiệm thành dữ liệu huấn luyện
                                train_data = []
                                for exp in experiences[:10]:  # Giới hạn số lượng experience để tránh quá tải
                                    s, a, r, ns, d = exp
                                    # Chuyển dữ liệu thành tensor
                                    s_tensor = torch.FloatTensor(s)
                                    a_tensor = torch.LongTensor([a[0]]) if isinstance(a, np.ndarray) else torch.LongTensor([a])
                                    r_tensor = torch.FloatTensor([r])
                                    
                                    train_data.append((s_tensor, a_tensor, r_tensor))
                                
                                # Thiết lập dữ liệu huấn luyện cho client
                                client.set_data(train_data)
                    
                    # Nếu có đủ shard hoạt động, tiến hành huấn luyện FL
                    if len(active_shards) >= 2:
                        # Chọn client tham gia vòng huấn luyện
                        selected_clients = fl_system.select_clients(fraction=0.8)
                        
                        # Huấn luyện mô hình cục bộ
                        for client_id in selected_clients:
                            client = fl_system.clients.get(client_id)
                            if client and client.local_data:
                                try:
                                    print(f"  Huấn luyện client {client_id}...")
                                    client.train_local_model()
                                except NotImplementedError:
                                    print(f"  Mô phỏng huấn luyện cho client {client_id}...")
                                    # Không cần làm gì vì dữ liệu đã được thiết lập
                        
                        # Thu thập mô hình đã huấn luyện
                        client_models = []
                        client_weights = []
                        
                        for client_id in selected_clients:
                            client = fl_system.clients.get(client_id)
                            if client:
                                client_models.append(client.get_model_params())
                                # Sử dụng điểm tin cậy của shard làm trọng số
                                try:
                                    # Lấy điểm tin cậy an toàn
                                    trust_score = htdcm.shard_trust_scores.get(client_id, 0.5)
                                except (AttributeError, IndexError):
                                    # Nếu không có shard_trust_scores, sử dụng giá trị mặc định
                                    trust_score = 0.5
                                client_weights.append(trust_score)
                        
                        # Tổng hợp mô hình
                        if len(client_models) >= 2:
                            print(f"  Tổng hợp mô hình từ {len(client_models)} clients...")
                            
                            # Chuẩn hóa trọng số
                            total_weight = sum(client_weights)
                            if total_weight > 0:
                                client_weights = [w / total_weight for w in client_weights]
                            else:
                                client_weights = [1.0 / len(client_weights)] * len(client_weights)
                            
                            # Tổng hợp mô hình
                            method = fl_system.method_mapping.get(fl_system.aggregation_method, 'weighted_average')
                            try:
                                global_model = fl_system.aggregation_manager.aggregate(
                                    method=method,
                                    params_list=client_models,
                                    weights=client_weights
                                )
                                
                                # Cập nhật mô hình toàn cục
                                fl_system.global_model.load_state_dict(global_model)
                                
                                # Tăng bộ đếm vòng FL
                                fl_system.round_counter += 1
                                print(f"  FL Round {fl_system.round_counter} hoàn thành")
                                
                                # Xóa bớt dữ liệu kinh nghiệm cũ để tránh quá tải
                                for i in range(len(shard_experiences)):
                                    if len(shard_experiences[i]) > 100:
                                        shard_experiences[i] = shard_experiences[i][-100:]
                            except Exception as e:
                                print(f"  Lỗi khi tổng hợp mô hình: {str(e)}")
                
                except Exception as e:
                    print(f"Lỗi khi thực hiện Federated Learning: {str(e)}")
            
            # Update for next step
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Update network state for router
            router.update_network_state(
                shard_congestion=env.shard_congestion,
                node_trust_scores=htdcm.get_node_trust_scores()
            )
        
        # Track performance
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        
        # Store metrics
        transaction_throughputs.append(info.get('successful_transactions', 0))
        episode_latencies = [info.get('latency', 0) for tx in env.transaction_pool if tx['status'] == 'completed']
        if episode_latencies:
            latencies.append(np.mean(episode_latencies))
        else:
            latencies.append(0)
        malicious_nodes = htdcm.identify_malicious_nodes()
        malicious_detections.append(len(malicious_nodes))
        
        # In training information
        log_interval = getattr(args, 'log_interval', 5)  # Mặc định là 5 nếu không có
        if steps % log_interval == 0:
            print(f"Episode {episode}/{args.episodes} - "
                 f"Reward: {episode_reward:.2f}, "
                 f"Avg Reward: {avg_reward:.2f}, "
                 f"Epsilon: {agent.epsilon:.4f}, "
                 f"Throughput: {transaction_throughputs[-1]}, "
                 f"Latency: {latencies[-1]:.2f}ms, "
                 f"Malicious Nodes: {len(malicious_nodes)}")
        
        # Save model
        if steps % 100 == 0 or steps == args.max_steps - 1:
            model_path = os.path.join(args.save_dir, f"dqn_model_ep{episode}.pth")
            agent.save(model_path)
            print(f"Saved model at: {model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "dqn_model_final.pth")
    agent.save(final_model_path)
    print(f"Saved final model at: {final_model_path}")
    
    # Save Federated Learning model (if any)
    if fl_system is not None:
        # Lưu mô hình toàn cục
        fl_model_path = os.path.join(args.save_dir, "federated_model_final.pth")
        fl_system.save_global_model(fl_model_path)
        print(f"Saved Federated Learning model at: {fl_model_path}")
        
        # Lưu lịch sử huấn luyện
        fl_history_path = os.path.join(args.save_dir, "federated_history.pt")
        fl_system.save_training_history(fl_history_path)
        
        # Tạo báo cáo hiệu suất FL
        if fl_system.round_counter > 0:
            client_summary = fl_system.get_client_performance_summary()
            print("\n=== FEDERATED LEARNING SUMMARY ===")
            print(f"Total rounds completed: {fl_system.round_counter}")
            print(f"Final global model loss: {fl_system.global_train_loss[-1]:.4f}" if fl_system.global_train_loss else "No training loss data")
            
            # Hiển thị thông tin về client
            print("\nClient performance:")
            for client_id, info in client_summary.items():
                trust = info['trust_score']
                part_count = info['participation_count']
                print(f"  Client {client_id}: Trust score: {trust:.2f}, Participated in {part_count} rounds")
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'transaction_throughputs': transaction_throughputs,
        'latencies': latencies,
        'malicious_detections': malicious_detections,
        'fl_metrics': {
            'rounds_completed': fl_system.round_counter if fl_system else 0,
            'global_loss': fl_system.global_train_loss if fl_system else [],
            'client_summary': fl_system.get_client_performance_summary() if fl_system else {}
        } if fl_system else None
    }

def evaluate_qtrust(env, agent, router, consensus, htdcm, fl_system, args):
    """
    Đánh giá hiệu suất của mô hình.
    
    Args:
        env: Môi trường blockchain
        agent: Agent DQN được huấn luyện
        router: Router định tuyến giao dịch
        consensus: Quản lý đồng thuận
        htdcm: Cơ chế tin cậy
        fl_system: Hệ thống federated learning
        args: Tham số dòng lệnh
        
    Returns:
        metrics: Các chỉ số hiệu suất
    """
    print("\nĐánh giá hiệu suất...")
    
    # Thiết lập các biến theo dõi
    rewards = []
    throughputs = []
    latencies = []
    energies = []
    securities = []
    cross_shard_ratios = []
    consensus_success_rates = {}
    protocol_usage = {}
    
    # FL evaluation metrics
    fl_prediction_accuracy = []
    fl_model_consistency = []
    
    # Thiết lập theo dõi cho số liệu giao thức
    for protocol_name in ["FastBFT", "PBFT", "RobustBFT", "LightBFT"]:
        consensus_success_rates[protocol_name] = []
        protocol_usage[protocol_name] = 0
        
    total_episodes = args.eval_episodes if hasattr(args, "eval_episodes") else 10
    print(f"Evaluating over {total_episodes} episodes...")
    
    # Danh sách các txs phức tạp để kiểm thử
    test_cases = []
    
    # Tạo từ 10-20 tình huống kiểm thử
    num_test_cases = random.randint(10, 20)
    for i in range(num_test_cases):
        # Tạo giao dịch mẫu với các giá trị khác nhau và loại khác nhau
        tx_type = random.choice(['intra_shard', 'cross_shard'])
        value = random.uniform(1.0, 100.0)
        source_shard = random.randint(0, env.num_shards - 1)
        dest_shard = random.randint(0, env.num_shards - 1) if tx_type == 'cross_shard' else source_shard
        
        test_cases.append({
            'type': tx_type,
            'value': value,
            'source_shard': source_shard,
            'destination_shard': dest_shard,
            'congestion': random.uniform(0.0, 1.0)
        })
    
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_throughput = 0
        episode_latency = []
        episode_energy = []
        episode_security = []
        episode_cross_shard_txs = 0
        episode_total_txs = 0
        
        # Track protocol usage within episode
        episode_protocol_use = {p: 0 for p in ["FastBFT", "PBFT", "RobustBFT", "LightBFT"]}
        episode_protocol_success = {p: [] for p in ["FastBFT", "PBFT", "RobustBFT", "LightBFT"]}
        
        # Store FL predictions for comparison if FL is enabled
        fl_predictions = []
        
        # Run full episode
        steps = 0
        while not done and steps < args.max_steps:
            # Get action from agent
            action = agent.act(state, eps=0.0)  # No exploration during evaluation
            
            # If FL system is available, get its prediction also
            if fl_system is not None and fl_system.global_model is not None:
                try:
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state).to(fl_system.device)
                    # Get raw predictions from FL model
                    with torch.no_grad():
                        fl_raw_output = fl_system.global_model(state_tensor)
                    
                    # Convert to action
                    fl_action_idx = torch.argmax(fl_raw_output).item()
                    shard_idx = fl_action_idx % env.num_shards
                    consensus_idx = (fl_action_idx // env.num_shards) % 3
                    fl_action = np.array([shard_idx, consensus_idx], dtype=np.int32)
                    
                    # Compare agent and FL predictions
                    same_shard = action[0] == fl_action[0]
                    same_consensus = action[1] == fl_action[1]
                    fl_predictions.append((same_shard, same_consensus))
                except Exception as e:
                    print(f"Error getting FL prediction: {str(e)}")
            
            # Execute in environment
            next_state, reward, done, info = env.step(action)
            
            # Compute metrics
            if isinstance(info, dict):
                # Track transaction performance
                txs_completed = info.get('successful_transactions', 0)
                episode_throughput += txs_completed
                
                tx_latency = info.get('latency', 0)
                if tx_latency > 0:
                    episode_latency.append(tx_latency)
                    
                tx_energy = info.get('energy_consumption', 0)
                if tx_energy > 0:
                    episode_energy.append(tx_energy)
                    
                tx_security = info.get('security_level', 0)
                if tx_security > 0:
                    episode_security.append(tx_security)
                
                # Count cross-shard transactions
                for tx in env.transaction_pool:
                    if tx.get('type') == 'cross_shard':
                        episode_cross_shard_txs += 1
                    episode_total_txs += 1
                
                # Track consensus protocol performance
                for tx in env.transaction_pool:
                    protocol_name = tx.get('consensus_protocol', 'unknown')
                    if protocol_name in episode_protocol_use:
                        episode_protocol_use[protocol_name] += 1
                        if tx.get('status') == 'completed':
                            episode_protocol_success[protocol_name].append(1)
                        else:
                            episode_protocol_success[protocol_name].append(0)
            
            # Update for next step
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Compute episode level metrics
        rewards.append(episode_reward)
        
        throughputs.append(episode_throughput / max(1, steps))
        
        if episode_latency:
            latencies.append(np.mean(episode_latency))
        else:
            latencies.append(0)
            
        if episode_energy:
            energies.append(np.mean(episode_energy))
        else:
            energies.append(0)
            
        if episode_security:
            securities.append(np.mean(episode_security))
        else:
            securities.append(0)
            
        cross_shard_ratio = episode_cross_shard_txs / max(1, episode_total_txs)
        cross_shard_ratios.append(cross_shard_ratio)
        
        # Process protocol performance
        for protocol in ["FastBFT", "PBFT", "RobustBFT", "LightBFT"]:
            if episode_protocol_use[protocol] > 0:
                success_rate = np.mean(episode_protocol_success[protocol])
                consensus_success_rates[protocol].append(success_rate)
                protocol_usage[protocol] += episode_protocol_use[protocol]
                
        # Calculate FL model accuracy if we have predictions
        if fl_predictions:
            shard_matches = sum(1 for p in fl_predictions if p[0])
            consensus_matches = sum(1 for p in fl_predictions if p[1])
            both_matches = sum(1 for p in fl_predictions if p[0] and p[1])
            
            shard_accuracy = shard_matches / len(fl_predictions)
            consensus_accuracy = consensus_matches / len(fl_predictions)
            action_accuracy = both_matches / len(fl_predictions)
            
            fl_prediction_accuracy.append({
                'shard_accuracy': shard_accuracy,
                'consensus_accuracy': consensus_accuracy,
                'action_accuracy': action_accuracy
            })
        
        # Print episode results
        print(f"Episode {episode+1}/{total_episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Throughput = {throughputs[-1]:.2f} tx/s, "
              f"Latency = {latencies[-1]:.2f} ms, "
              f"Energy = {energies[-1]:.2f} mJ/tx, "
              f"Security = {securities[-1]:.2f}, "
              f"Cross-shard ratio = {cross_shard_ratio:.2f}")
    
    # Evaluate FL model on specific test cases if available
    if fl_system is not None and fl_system.global_model is not None and test_cases:
        print("\n=== Federated Learning Model Evaluation ===")
        fl_test_results = []
        
        # Test FL model on each test case
        for i, test_case in enumerate(test_cases):
            # Create input state from test case
            test_state = np.zeros(env.observation_space.shape)
            
            # Add relevant features from test case to test state
            # This is a simplification - in a real system, you'd create a proper state
            test_state[0] = test_case['value'] / 100.0  # Normalize value
            test_state[1] = 1.0 if test_case['type'] == 'cross_shard' else 0.0
            test_state[2] = test_case['source_shard'] / max(1, env.num_shards - 1)
            test_state[3] = test_case['destination_shard'] / max(1, env.num_shards - 1)
            test_state[4] = test_case['congestion']
            
            # Get FL model prediction
            try:
                state_tensor = torch.FloatTensor(test_state).to(fl_system.device)
                with torch.no_grad():
                    fl_output = fl_system.global_model(state_tensor)
                
                # Get best action
                fl_action_idx = torch.argmax(fl_output).item()
                shard_idx = fl_action_idx % env.num_shards
                consensus_idx = (fl_action_idx // env.num_shards) % 3
                
                # Map consensus index to protocol
                protocols = ["FastBFT", "PBFT", "RobustBFT"]
                protocol = protocols[consensus_idx] if consensus_idx < len(protocols) else "Unknown"
                
                # Store test result
                fl_test_results.append({
                    'test_case': test_case,
                    'predicted_shard': shard_idx,
                    'predicted_protocol': protocol
                })
                
                # Print some details
                if i < 5:  # Print first 5 test cases only
                    print(f"Test {i+1}: {test_case['type']} tx, value={test_case['value']:.2f}, "
                          f"source={test_case['source_shard']}, dest={test_case['destination_shard']} -> "
                          f"FL model suggests shard {shard_idx} with {protocol}")
            except Exception as e:
                print(f"Error evaluating FL model on test case {i+1}: {str(e)}")
        
        # Simple evaluation - are predictions consistent with expectations?
        # For example, high value transactions should use more secure protocols
        high_value_cases = [r for r in fl_test_results if r['test_case']['value'] > 50.0]
        if high_value_cases:
            secure_protocols = ["PBFT", "RobustBFT"]
            secure_choices = sum(1 for r in high_value_cases if r['predicted_protocol'] in secure_protocols)
            security_consistency = secure_choices / len(high_value_cases)
            fl_model_consistency.append(('high_value_security', security_consistency))
            
            print(f"\nFL model security consistency on high-value transactions: {security_consistency:.2f}")
        
        # Check if cross-shard txs are routed through appropriate shards
        cross_shard_cases = [r for r in fl_test_results if r['test_case']['type'] == 'cross_shard']
        if cross_shard_cases:
            # Check if predicted shard matches source or destination
            matching_shard = sum(1 for r in cross_shard_cases 
                               if r['predicted_shard'] == r['test_case']['source_shard'] or 
                                  r['predicted_shard'] == r['test_case']['destination_shard'])
            shard_consistency = matching_shard / len(cross_shard_cases)
            fl_model_consistency.append(('cross_shard_routing', shard_consistency))
            
            print(f"FL model routing consistency on cross-shard transactions: {shard_consistency:.2f}")
    
    # Calculate average metrics
    avg_reward = np.mean(rewards)
    avg_throughput = np.mean(throughputs)
    avg_latency = np.mean(latencies)
    avg_energy = np.mean(energies)
    avg_security = np.mean(securities)
    avg_cross_shard_ratio = np.mean(cross_shard_ratios)
    
    # Calculate consensus protocol performance
    protocol_performance = {}
    for protocol in ["FastBFT", "PBFT", "RobustBFT", "LightBFT"]:
        if consensus_success_rates[protocol]:
            success_rate = np.mean(consensus_success_rates[protocol])
            usage_percent = protocol_usage[protocol] / max(1, sum(protocol_usage.values())) * 100
            protocol_performance[protocol] = {
                'success_rate': success_rate,
                'usage_percent': usage_percent
            }
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average throughput: {avg_throughput:.2f} tx/s")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Average energy consumption: {avg_energy:.2f} mJ/tx")
    print(f"Average security level: {avg_security:.2f}")
    print(f"Average cross-shard ratio: {avg_cross_shard_ratio:.2f}")
    
    print("\nConsensus Protocol Performance:")
    for protocol, perf in protocol_performance.items():
        print(f"  {protocol}: Success rate = {perf['success_rate']:.2f}, Used in {perf['usage_percent']:.2f}% of transactions")
    
    # Prepare FL metrics if available
    fl_evaluation = None
    if fl_system and (fl_prediction_accuracy or fl_model_consistency):
        fl_evaluation = {
            'prediction_accuracy': fl_prediction_accuracy,
            'model_consistency': dict(fl_model_consistency),
            'test_results': fl_test_results if 'fl_test_results' in locals() else []
        }
        
        if fl_prediction_accuracy:
            avg_accs = [item['action_accuracy'] for item in fl_prediction_accuracy]
            print("\nFederated Learning Model Performance:")
            print(f"  Shard selection accuracy: {np.mean(avg_accs)*100:.2f}%")
    
    # Return comprehensive metrics
    return {
        'rewards': rewards,
        'avg_reward': avg_reward,
        'throughputs': throughputs,
        'avg_throughput': avg_throughput,
        'latencies': latencies,
        'avg_latency': avg_latency,
        'energies': energies,
        'avg_energy': avg_energy,
        'securities': securities,
        'avg_security': avg_security,
        'cross_shard_ratios': cross_shard_ratios,
        'avg_cross_shard_ratio': avg_cross_shard_ratio,
        'protocol_performance': protocol_performance,
        'fl_evaluation': fl_evaluation
    }

# Hàm hỗ trợ điều chỉnh kích thước state
def adapt_state_for_model(state, model_state_size):
    """Điều chỉnh kích thước state để phù hợp với model."""
    if len(state) > model_state_size:
        # Nếu state lớn hơn, lấy phần đầu tiên phù hợp với model
        return state[:model_state_size]
    elif len(state) < model_state_size:
        # Nếu state nhỏ hơn, mở rộng với các giá trị 0
        padded_state = np.zeros(model_state_size)
        padded_state[:len(state)] = state
        return padded_state
    else:
        return state

def plot_results(metrics, args, mode='train'):
    """
    Vẽ biểu đồ kết quả huấn luyện hoặc đánh giá.
    
    Args:
        metrics: Dictionary chứa các metrics
        args: Tham số dòng lệnh
        mode: 'train' hoặc 'eval'
    """
    print(f"Vẽ đồ thị kết quả {'huấn luyện' if mode == 'train' else 'đánh giá'}...")
    
    # Tạo thư mục cho các biểu đồ
    plots_dir = os.path.join(args.save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Đảm bảo sử dụng font tiếng Việt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Arial']
    
    # 1. Biểu đồ Reward theo thời gian
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['rewards'], label='Episode Reward', alpha=0.4, color='#1f77b4')
    
    # Đường trung bình động
    window_size = max(1, len(metrics['rewards']) // 5)
    if len(metrics['rewards']) > 1:
        smoothed_rewards = np.convolve(metrics['rewards'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(metrics['rewards'])), smoothed_rewards, 
                color='blue', label=f'Moving Avg (window={window_size})')
    
    plt.title('Reward theo Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{mode}_rewards.png'), dpi=300, bbox_inches='tight')
    
    if mode == 'train':
        # 2. Biểu đồ kết hợp Throughput, Latency và Energy Consumption
        plt.figure(figsize=(12, 8))
        
        if 'throughputs' in metrics and len(metrics['throughputs']) > 0:
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(metrics['throughputs'], color='green', alpha=0.6)
            ax1.set_title('Throughput (tx/s)')
            ax1.grid(alpha=0.3)
            
            if len(metrics['throughputs']) > 1:
                smoothed = np.convolve(metrics['throughputs'], np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(metrics['throughputs'])), smoothed, color='darkgreen')
        
        if 'latencies' in metrics and len(metrics['latencies']) > 0:
            ax2 = plt.subplot(3, 1, 2)
            ax2.plot(metrics['latencies'], color='red', alpha=0.6)
            ax2.set_title('Latency (ms)')
            ax2.grid(alpha=0.3)
            
            if len(metrics['latencies']) > 1:
                smoothed = np.convolve(metrics['latencies'], np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(metrics['latencies'])), smoothed, color='darkred')
        
        if 'energies' in metrics and len(metrics['energies']) > 0:
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(metrics['energies'], color='orange', alpha=0.6)
            ax3.set_title('Energy Consumption (mJ/tx)')
            ax3.grid(alpha=0.3)
            
            if len(metrics['energies']) > 1:
                smoothed = np.convolve(metrics['energies'], np.ones(window_size)/window_size, mode='valid')
                ax3.plot(range(window_size-1, len(metrics['energies'])), smoothed, color='darkorange')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{mode}_performance.png'), dpi=300, bbox_inches='tight')
        
        # 3. Biểu đồ Security và Cross-shard ratio
        plt.figure(figsize=(10, 8))
        
        if 'securities' in metrics and len(metrics['securities']) > 0:
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(metrics['securities'], color='purple', alpha=0.6)
            ax1.set_title('Security Score')
            ax1.set_ylim([0, 1])
            ax1.grid(alpha=0.3)
            
            if len(metrics['securities']) > 1:
                smoothed = np.convolve(metrics['securities'], np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(metrics['securities'])), smoothed, color='purple')
        
        if 'cross_shard_ratios' in metrics and len(metrics['cross_shard_ratios']) > 0:
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(metrics['cross_shard_ratios'], color='blue', alpha=0.6)
            ax2.set_title('Cross-shard Transaction Ratio')
            ax2.set_ylim([0, 1])
            ax2.grid(alpha=0.3)
            
            if len(metrics['cross_shard_ratios']) > 1:
                smoothed = np.convolve(metrics['cross_shard_ratios'], np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(metrics['cross_shard_ratios'])), smoothed, color='blue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{mode}_security.png'), dpi=300, bbox_inches='tight')
    
    # Đóng tất cả các biểu đồ để giải phóng bộ nhớ
    plt.close('all')
    
    print(f"Đã lưu biểu đồ kết quả vào {plots_dir}")

def create_blockchain_network(num_shards, nodes_per_shard):
    """
    Initialize blockchain network with a specific number of shards and nodes.
    
    Args:
        num_shards: Number of shards
        nodes_per_shard: Number of nodes in each shard
        
    Returns:
        tuple: (network graph, node to shard mapping, list of shards)
    """
    # Initialize graph
    G = nx.Graph()
    total_nodes = num_shards * nodes_per_shard
    
    # Create nodes
    for i in range(total_nodes):
        G.add_node(i, trust_score=0.7)
    
    # Create connections between nodes
    # Create list of shards, each shard contains node IDs
    shards = []
    node_to_shard = {}
    
    for shard_id in range(num_shards):
        shard_nodes = list(range(shard_id * nodes_per_shard, (shard_id + 1) * nodes_per_shard))
        shards.append(shard_nodes)
        
        # Map nodes to shards
        for node_id in shard_nodes:
            node_to_shard[node_id] = shard_id
        
        # Intra-shard connections (each node connects to all other nodes in the shard)
        for i in range(len(shard_nodes)):
            for j in range(i + 1, len(shard_nodes)):
                G.add_edge(shard_nodes[i], shard_nodes[j])
    
    # Inter-shard connections (each shard has a certain number of connections to other shards)
    nodes_per_connection = min(3, nodes_per_shard // 2)  # Number of connections between shards
    
    for i in range(num_shards):
        for j in range(i + 1, num_shards):
            for _ in range(nodes_per_connection):
                node_i = random.choice(shards[i])
                node_j = random.choice(shards[j])
                G.add_edge(node_i, node_j)
    
    # Add shard information to each node
    for shard_id, shard in enumerate(shards):
        for node_id in shard:
            G.nodes[node_id]['shard_id'] = shard_id
    
    return G, node_to_shard, shards

def main():
    """Main function."""
    args = parse_args()
    
    # Setup simulation environment and components
    env, router, htdcm, ac_manager = setup_simulation(args)
    
    # Print settings
    print("\n============================================================")
    print("QTrust: Optimized Blockchain Sharding with Deep Reinforcement Learning")
    print("============================================================\n")
    print(f"Number of shards: {args.num_shards}")
    print(f"Nodes per shard: {args.nodes_per_shard}")
    
    if args.enable_adaptive_pos:
        print(f"Adaptive PoS: Enabled")
        print(f"  - Active validator ratio: {args.active_validator_ratio}")
        print(f"  - Rotation period: {args.pos_rotation_period} rounds")
    else:
        print("Adaptive PoS: Disabled")
        
    if args.enable_lightweight_crypto:
        print(f"Lightweight Cryptography: Enabled")
        print(f"  - Energy optimization level: {args.energy_optimization_level}")
    else:
        print("Lightweight Cryptography: Disabled")
    
    print(f"Steps per episode: {args.max_steps}")
    print(f"Number of episodes: {args.episodes}")
    
    print(f"\n=== Q-TRUST: Optimized Blockchain Sharding System with Deep Reinforcement Learning ===")
    print(f"Device: {args.device}")
    print(f"Number of shards: {args.num_shards}")
    print(f"Nodes per shard: {args.nodes_per_shard}")
    print(f"Mode: {'EVALUATION' if args.eval else 'TRAINING'}")
    
    print("Initializing simulation environment...")
    print("Initializing QTrust system components...")
    
    # Setup Federated Learning if enabled
    if args.enable_federated:
        print("Khởi tạo Federated Learning...")
        fl_system = setup_federated_learning(env, args, htdcm)
    else:
        fl_system = None
    
    # Initialize DQN Agent
    agent = setup_dqn_agent(env, args)
    
    if args.enable_adaptive_pos:
        print("Initializing Adaptive PoS for each shard...")
        # Khởi tạo PoSManager cho từng shard
        for shard_id in range(env.num_shards):
            ac_manager.pos_managers[shard_id] = AdaptivePoSManager(
                num_validators=env.num_nodes_per_shard,
                active_validator_ratio=args.active_validator_ratio,
                rotation_period=args.pos_rotation_period,
                energy_optimization_level=args.energy_optimization_level,
                enable_smart_energy_management=True
            )
        print(f"PoS initialization completed for {env.num_shards} shards")
    
    print("All components have been successfully activated!")
    print("\n=== ACTIVATED COMPONENTS ===")
    print("1. Hierarchical Trust-based Data Center Mechanism (HTDCM)")
    print("2. Adaptive Consensus (AC)")
    if fl_system:
        print("3. Federated Learning Manager (FL)")
    print("5. MAD-RAPID Router")
    print(f"6. Blockchain Environment ({env.num_shards} shards, {env.num_nodes_per_shard} nodes/shard)")
    
    print("\nStarting full simulation...\n")
    
    # Run evaluation or training based on args
    if args.eval:
        eval_results = evaluate_qtrust(env, agent, router, ac_manager, htdcm, fl_system, args)
        
        # Save evaluation results
        eval_path = os.path.join(args.save_dir, 'evaluation_results.json')
        
        # Convert numpy arrays and other non-serializable objects to lists
        serializable_results = {}
        for key, value in eval_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and all(isinstance(x, np.ndarray) for x in value):
                serializable_results[key] = [x.tolist() for x in value]
            elif isinstance(value, (int, float, str, bool)):
                serializable_results[key] = value
            elif value is None:
                serializable_results[key] = None
            else:
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[key] = value
                except (TypeError, OverflowError):
                    print(f"Warning: Could not serialize {key}, converting to string representation")
                    serializable_results[key] = str(value)
        
        with open(eval_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Evaluation results saved to {eval_path}")
        
        # Hiển thị thông tin chi tiết về Federated Learning nếu được bật
        if args.enable_federated and fl_system and 'fl_evaluation' in eval_results and eval_results['fl_evaluation']:
            fl_eval = eval_results['fl_evaluation']
            
            print("\n=== FEDERATED LEARNING DETAILED EVALUATION ===")
            
            # Hiển thị độ chính xác trung bình
            if 'prediction_accuracy' in fl_eval and fl_eval['prediction_accuracy']:
                avg_accs = [item['action_accuracy'] for item in fl_eval['prediction_accuracy']]
                print(f"FL model achieves {np.mean(avg_accs)*100:.2f}% agreement with DQN model")
            
            # Hiển thị độ nhất quán trong các quyết định
            if 'model_consistency' in fl_eval:
                for test_name, score in fl_eval['model_consistency'].items():
                    print(f"Consistency test '{test_name}': {score*100:.2f}%")
            
            # Lưu báo cáo riêng về FL
            fl_report_path = os.path.join(args.save_dir, 'fl_evaluation_report.json')
            with open(fl_report_path, 'w') as f:
                json.dump(fl_eval, f, indent=2)
                
            print(f"Detailed FL evaluation saved to {fl_report_path}")
            
            # Lưu mô hình FL
            if hasattr(fl_system, 'global_model'):
                fl_model_path = os.path.join(args.save_dir, 'evaluated_fl_model.pth')
                fl_system.save_global_model(fl_model_path)
                print(f"FL model saved to {fl_model_path}")
    else:
        train_results = train_qtrust(env, agent, router, ac_manager, htdcm, fl_system, args)

if __name__ == "__main__":
    main() 