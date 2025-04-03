import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym

from qtrust.federated.federated_rl import FederatedRL, FRLClient
from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.networks import QNetwork

# Môi trường đơn giản cho kiểm thử
class SimpleEnv:
    def __init__(self):
        self.state = np.zeros(4)
        self.done = False
        self.steps = 0
        self.max_steps = 10
        
    def reset(self):
        self.state = np.random.rand(4)
        self.done = False
        self.steps = 0
        return self.state
    
    def step(self, action):
        # Action từ 0-1
        if action == 0:
            reward = 0.1
        else:
            reward = 0.2 if np.sum(self.state) > 2.0 else -0.1
            
        # Cập nhật trạng thái
        self.state = np.random.rand(4)
        
        # Kiểm tra kết thúc
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            
        return self.state, reward, self.done, {}

class TestFederatedRL(unittest.TestCase):
    
    def setUp(self):
        # Đặt seed để đảm bảo kết quả tái lập
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Tham số cho môi trường và mạng
        self.state_size = 4
        self.action_size = 2
        self.batch_size = 8
        self.buffer_size = 100
        
        # Tạo mô hình toàn cục
        self.global_model = QNetwork(
            state_dim=(self.state_size,), 
            action_dim=(self.action_size,),
            hidden_dims=[8, 8],
            dueling=False
        )
        
        # Tạo FederatedRL
        self.fed_rl = FederatedRL(
            global_model=self.global_model,
            aggregation_method='fedavg',
            min_clients_per_round=2,
            min_samples_per_client=5,
            device='cpu'
        )
        
        # Môi trường toàn cục để đánh giá
        self.global_env = SimpleEnv()
        self.fed_rl.set_global_environment(self.global_env)
    
    def test_client_creation(self):
        # Tạo client
        agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=0,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device='cpu'
        )
        
        client = FRLClient(
            client_id=1,
            agent=agent,
            model=agent.qnetwork_local,
            local_epochs=2,
            batch_size=self.batch_size
        )
        
        self.fed_rl.add_client(client)
        self.assertIn(1, self.fed_rl.clients)
        
        # Thiết lập môi trường cho client
        env = SimpleEnv()
        client.set_environment(env)
        
        # Thu thập kinh nghiệm
        experiences = client.collect_experiences(num_steps=20, epsilon=0.5)
        self.assertGreaterEqual(len(client.local_experiences), 20)
    
    def test_training_round(self):
        # Tạo và thêm 3 clients
        for i in range(3):
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                seed=i,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                device='cpu'
            )
            
            client = FRLClient(
                client_id=i,
                agent=agent,
                model=agent.qnetwork_local,
                local_epochs=2,
                batch_size=self.batch_size
            )
            
            # Thiết lập môi trường cho client
            env = SimpleEnv()
            client.set_environment(env)
            
            self.fed_rl.add_client(client)
        
        # Thực hiện một vòng huấn luyện
        metrics = self.fed_rl.train_round(
            round_num=1,
            client_fraction=0.8,
            steps_per_client=20,
            exploration_epsilon=0.5
        )
        
        # Kiểm tra các metrics
        self.assertEqual(metrics['round'], 1)
        self.assertGreaterEqual(len(metrics['clients']), 2)
        self.assertIsNotNone(metrics['avg_train_loss'])
        
        # Kiểm tra reward toàn cục
        self.assertIsNotNone(metrics['global_reward'])
    
    def test_multiple_rounds(self):
        # Tạo và thêm 3 clients
        for i in range(3):
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                seed=i,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                device='cpu'
            )
            
            client = FRLClient(
                client_id=i,
                agent=agent,
                model=agent.qnetwork_local,
                local_epochs=2,
                batch_size=self.batch_size
            )
            
            # Thiết lập môi trường cho client
            env = SimpleEnv()
            client.set_environment(env)
            
            self.fed_rl.add_client(client)
        
        # Huấn luyện 3 vòng
        result = self.fed_rl.train(
            num_rounds=3,
            client_fraction=0.8,
            steps_per_client=20,
            exploration_schedule=lambda r: 0.5,
            verbose=True
        )
        
        # Kiểm tra kết quả
        self.assertEqual(result['rounds_completed'], 3)
        self.assertIsNotNone(result['best_reward'])
        self.assertEqual(len(result['train_loss_history']), 3)
        self.assertEqual(len(result['round_metrics']), 3)
    
    def test_privacy_preserving(self):
        # Tạo FederatedRL với tính năng bảo vệ quyền riêng tư
        private_fed_rl = FederatedRL(
            global_model=self.global_model,
            aggregation_method='fedavg',
            min_clients_per_round=2,
            min_samples_per_client=5,
            device='cpu',
            privacy_preserving=True,
            privacy_epsilon=0.1
        )
        
        # Tạo và thêm 3 clients
        for i in range(3):
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                seed=i,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                device='cpu'
            )
            
            client = FRLClient(
                client_id=i,
                agent=agent,
                model=agent.qnetwork_local,
                local_epochs=2,
                batch_size=self.batch_size
            )
            
            # Thiết lập môi trường cho client
            env = SimpleEnv()
            client.set_environment(env)
            
            private_fed_rl.add_client(client)
        
        # Thực hiện một vòng huấn luyện
        private_fed_rl.set_global_environment(self.global_env)
        metrics = private_fed_rl.train_round(
            round_num=1,
            client_fraction=0.8,
            steps_per_client=20,
            exploration_epsilon=0.5
        )
        
        # Kiểm tra rằng huấn luyện đã diễn ra
        self.assertEqual(metrics['round'], 1)
        self.assertIsNotNone(metrics['avg_train_loss'])

if __name__ == '__main__':
    unittest.main() 