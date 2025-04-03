"""
Energy optimization simulation module for the QTrust blockchain system.

This module provides a simulation environment to test the energy efficiency of the Adaptive PoS (Proof of Stake) 
consensus mechanism. It compares the Adaptive PoS approach with standard consensus mechanisms in terms of 
energy consumption, transaction success rate, and validator activity over multiple simulation rounds.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any
import os
import sys
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.consensus.adaptive_pos import AdaptivePoSManager


class EnergySimulation:
    """
    Simulation to test energy saving efficiency of Adaptive PoS.
    """
    def __init__(self, 
                 simulation_rounds: int = 1000,
                 num_shards: int = 5,
                 validators_per_shard: int = 15,
                 active_ratio_pos: float = 0.7,
                 rotation_period: int = 50,
                 transaction_rate: float = 10.0,
                 plot_results: bool = True,
                 save_dir: str = "results"):
        """
        Initialize simulation.
        
        Args:
            simulation_rounds: Number of simulation rounds
            num_shards: Number of shards
            validators_per_shard: Number of validators per shard
            active_ratio_pos: Ratio of active validators in Adaptive PoS
            rotation_period: Validator rotation period
            transaction_rate: Average transaction rate (transactions/round)
            plot_results: Whether to plot results
            save_dir: Directory to save results
        """
        self.simulation_rounds = simulation_rounds
        self.num_shards = num_shards
        self.validators_per_shard = validators_per_shard
        self.active_ratio_pos = active_ratio_pos
        self.rotation_period = rotation_period
        self.transaction_rate = transaction_rate
        self.plot_results = plot_results
        self.save_dir = save_dir
        
        # Initialize consensus mechanisms
        self.adaptive_consensus = AdaptiveConsensus(
            num_validators_per_shard=validators_per_shard,
            enable_adaptive_pos=True,
            active_validator_ratio=active_ratio_pos,
            rotation_period=rotation_period
        )
        
        self.standard_consensus = AdaptiveConsensus(
            num_validators_per_shard=validators_per_shard,
            enable_adaptive_pos=False
        )
        
        # Initialize trust scores for each validator in each shard
        self.trust_scores = {}
        for shard in range(num_shards):
            for validator in range(1, validators_per_shard + 1):
                # Validator ID: shard_id * 100 + validator_id
                validator_id = shard * 100 + validator
                self.trust_scores[validator_id] = 0.5 + random.random() * 0.5  # 0.5-1.0
        
        # Initialize congestion levels for each shard
        self.congestion_levels = {shard: 0.2 + random.random() * 0.3 for shard in range(num_shards)}
        
        # Variables to track results
        self.results = {
            "rounds": [],
            "adaptive_pos_energy": [],
            "standard_energy": [],
            "adaptive_pos_success_rate": [],
            "standard_success_rate": [],
            "energy_saved": [],
            "rotations": [],
            "active_validators": []
        }
        
        # Create results directory if it doesn't exist
        if plot_results and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def simulate_transaction_batch(self, consensus, shard_id: int, num_transactions: int) -> Dict[str, Any]:
        """
        Simulate a batch of transactions on a shard.
        
        Args:
            consensus: AdaptiveConsensus object
            shard_id: ID of the shard
            num_transactions: Number of transactions to process
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        # Filter trust scores for validators in this shard
        shard_validators = {v_id: score for v_id, score in self.trust_scores.items() 
                           if v_id // 100 == shard_id}
        
        total_energy = 0.0
        successful_txs = 0
        
        for _ in range(num_transactions):
            # Create a random transaction
            tx_value = random.expovariate(1.0 / 20.0)  # Average transaction value 20
            is_cross_shard = random.random() < 0.3  # 30% are cross-shard transactions
            
            # Execute consensus
            result, protocol, latency, energy = consensus.execute_consensus(
                transaction_value=tx_value,
                congestion=self.congestion_levels[shard_id],
                trust_scores=shard_validators,
                network_stability=0.7,
                cross_shard=is_cross_shard,
                shard_id=shard_id
            )
            
            # Record results
            total_energy += energy
            if result:
                successful_txs += 1
        
        return {
            "total_energy": total_energy,
            "successful_txs": successful_txs,
            "total_txs": num_transactions,
            "success_rate": successful_txs / num_transactions if num_transactions > 0 else 0
        }
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run simulation for the entire system.
        
        Returns:
            Dict[str, Any]: Simulation results
        """
        print(f"Starting simulation with {self.simulation_rounds} rounds...")
        
        for round_num in range(1, self.simulation_rounds + 1):
            # Update congestion levels (varies over time)
            for shard in range(self.num_shards):
                # 10% fluctuation per round
                delta = (random.random() - 0.5) * 0.1
                self.congestion_levels[shard] = max(0.1, min(0.9, self.congestion_levels[shard] + delta))
            
            # Calculate transactions per shard (based on congestion levels)
            transactions_per_shard = {
                shard: int(self.transaction_rate * (0.8 + self.congestion_levels[shard]))
                for shard in range(self.num_shards)
            }
            
            # Simulate with Adaptive PoS
            adaptive_pos_results = {
                shard: self.simulate_transaction_batch(
                    self.adaptive_consensus, shard, transactions_per_shard[shard]
                ) for shard in range(self.num_shards)
            }
            
            # Simulate with standard consensus (not using Adaptive PoS)
            standard_results = {
                shard: self.simulate_transaction_batch(
                    self.standard_consensus, shard, transactions_per_shard[shard]
                ) for shard in range(self.num_shards)
            }
            
            # Calculate total energy and success rates
            adaptive_pos_energy = sum(r["total_energy"] for r in adaptive_pos_results.values())
            standard_energy = sum(r["total_energy"] for r in standard_results.values())
            
            adaptive_pos_success = sum(r["successful_txs"] for r in adaptive_pos_results.values())
            standard_success = sum(r["successful_txs"] for r in standard_results.values())
            
            total_txs = sum(r["total_txs"] for r in adaptive_pos_results.values())
            
            adaptive_pos_success_rate = adaptive_pos_success / total_txs if total_txs > 0 else 0
            standard_success_rate = standard_success / total_txs if total_txs > 0 else 0
            
            # Get statistics from Adaptive PoS
            pos_stats = self.adaptive_consensus.get_pos_statistics()
            
            # Calculate total active validators across all shards
            active_validators = 0
            for shard_id in range(self.num_shards):
                if shard_id in pos_stats["shard_stats"]:
                    active_validators += pos_stats["shard_stats"][shard_id]["validators"]["active_validators"]
            
            # Save results
            self.results["rounds"].append(round_num)
            self.results["adaptive_pos_energy"].append(adaptive_pos_energy)
            self.results["standard_energy"].append(standard_energy)
            self.results["adaptive_pos_success_rate"].append(adaptive_pos_success_rate)
            self.results["standard_success_rate"].append(standard_success_rate)
            self.results["energy_saved"].append(pos_stats["total_energy_saved"])
            self.results["rotations"].append(pos_stats["total_rotations"])
            self.results["active_validators"].append(active_validators)
            
            # Print progress
            if round_num % 100 == 0 or round_num == 1:
                energy_saving = (1 - adaptive_pos_energy / standard_energy) * 100 if standard_energy > 0 else 0
                print(f"Round {round_num}/{self.simulation_rounds}: Energy saving {energy_saving:.2f}%, "
                      f"Active validators: {active_validators}/{self.num_shards * self.validators_per_shard}")
        
        # Plot results
        if self.plot_results:
            self.plot_simulation_results()
        
        # Calculate final results
        total_adaptive_energy = sum(self.results["adaptive_pos_energy"])
        total_standard_energy = sum(self.results["standard_energy"])
        energy_saving_percent = (1 - total_adaptive_energy / total_standard_energy) * 100 if total_standard_energy > 0 else 0
        
        avg_adaptive_success = np.mean(self.results["adaptive_pos_success_rate"]) * 100
        avg_standard_success = np.mean(self.results["standard_success_rate"]) * 100
        
        final_results = {
            "total_rounds": self.simulation_rounds,
            "total_adaptive_energy": total_adaptive_energy,
            "total_standard_energy": total_standard_energy,
            "energy_saving_percent": energy_saving_percent,
            "avg_adaptive_success": avg_adaptive_success,
            "avg_standard_success": avg_standard_success,
            "total_pos_energy_saved": self.results["energy_saved"][-1],
            "total_rotations": self.results["rotations"][-1],
            "final_active_validators": self.results["active_validators"][-1]
        }
        
        print("\nSimulation Results:")
        print(f"Total rounds: {final_results['total_rounds']}")
        print(f"Total energy (Adaptive PoS): {final_results['total_adaptive_energy']:.2f}")
        print(f"Total energy (Standard): {final_results['total_standard_energy']:.2f}")
        print(f"Energy saving: {final_results['energy_saving_percent']:.2f}%")
        print(f"Success rate (Adaptive PoS): {final_results['avg_adaptive_success']:.2f}%")
        print(f"Success rate (Standard): {final_results['avg_standard_success']:.2f}%")
        print(f"Total validator rotations: {final_results['total_rotations']}")
        print(f"Final active validators: {final_results['final_active_validators']}")
        
        return final_results
    
    def plot_simulation_results(self):
        """Plot simulation results."""
        # Create figure with 4 subplots
        plt.figure(figsize=(16, 14))
        
        # 1. Energy consumption comparison chart
        plt.subplot(2, 2, 1)
        plt.plot(self.results["rounds"], self.results["adaptive_pos_energy"], 
                 label="Adaptive PoS", color="green")
        plt.plot(self.results["rounds"], self.results["standard_energy"], 
                 label="Standard", color="red")
        plt.xlabel("Simulation round")
        plt.ylabel("Energy consumption")
        plt.title("Energy consumption comparison")
        plt.legend()
        plt.grid(True)
        
        # 2. Energy saved chart
        plt.subplot(2, 2, 2)
        plt.plot(self.results["rounds"], self.results["energy_saved"], color="blue")
        plt.xlabel("Simulation round")
        plt.ylabel("Energy saved")
        plt.title("Energy saved (Adaptive PoS)")
        plt.grid(True)
        
        # 3. Validator rotations chart
        plt.subplot(2, 2, 3)
        plt.plot(self.results["rounds"], self.results["rotations"], color="purple")
        plt.xlabel("Simulation round")
        plt.ylabel("Number of rotations")
        plt.title("Total validator rotations")
        plt.grid(True)
        
        # 4. Success rate chart
        plt.subplot(2, 2, 4)
        plt.plot(self.results["rounds"], [rate * 100 for rate in self.results["adaptive_pos_success_rate"]], 
                 label="Adaptive PoS", color="green")
        plt.plot(self.results["rounds"], [rate * 100 for rate in self.results["standard_success_rate"]], 
                 label="Standard", color="red")
        plt.xlabel("Simulation round")
        plt.ylabel("Success rate (%)")
        plt.title("Success rate comparison")
        plt.legend()
        plt.grid(True)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "energy_optimization_results.png"))
        plt.close()
        
        print(f"Results chart saved to {self.save_dir}/energy_optimization_results.png")


def main():
    """Run simulation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Energy optimization simulation for QTrust')
    parser.add_argument('--rounds', type=int, default=200, help='Number of simulation rounds')
    parser.add_argument('--shards', type=int, default=2, help='Number of shards')
    parser.add_argument('--validators', type=int, default=10, help='Validators per shard')
    parser.add_argument('--active-ratio', type=float, default=0.7, help='Active validator ratio')
    parser.add_argument('--rotation-period', type=int, default=20, help='Validator rotation period')
    parser.add_argument('--tx-rate', type=float, default=10.0, help='Transaction rate')
    parser.add_argument('--no-plot', action='store_true', help='Do not plot results')
    parser.add_argument('--save-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Run simulation
    sim = EnergySimulation(
        simulation_rounds=args.rounds,
        num_shards=args.shards,
        validators_per_shard=args.validators,
        active_ratio_pos=args.active_ratio,
        rotation_period=args.rotation_period,
        transaction_rate=args.tx_rate,
        plot_results=not args.no_plot,
        save_dir=args.save_dir
    )
    
    results = sim.run_simulation()
    return results


if __name__ == "__main__":
    main() 