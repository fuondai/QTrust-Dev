#!/bin/bash

# Cache performance evaluation script for QTrust project
# Runs each agent and records results for comparison
# 
# This script evaluates the impact of caching on different RL agent performance
# in the QTrust blockchain system. It runs DQN, Rainbow DQN, and Actor-Critic
# agents both with and without caching enabled, and generates comparison reports.

echo "========== QTRUST CACHING PERFORMANCE EVALUATION =========="
echo "Start time: $(date)"
echo

# Check Python availability and platform
python_cmd="python"
# On Windows, try to use py launcher if available
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    if command -v py &> /dev/null; then
        python_cmd="py -3.10"
    fi
else
    # On Linux/Mac, use python3 if available
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    fi
fi


echo "Using Python command: $python_cmd"

# Check if the test script exists
test_script="test_caching.py"
if [ ! -f "$test_script" ]; then
    echo "Error: Test script $test_script not found!"
    echo "Please make sure you are running this script from the correct directory."
    echo "Current directory: $(pwd)"
    echo "Looking for: $(pwd)/$test_script"
    exit 1
fi

# Create directory for results
results_dir="cache_results"
mkdir -p "$results_dir"

# Run test with DQN Agent
echo "===== DQN Agent Evaluation ====="
echo "Running with caching..."
$python_cmd "$test_script" --agent dqn --episodes 10 > "$results_dir/dqn_with_cache.log"
echo "Running without caching..."
$python_cmd "$test_script" --agent dqn --episodes 10 --disable-cache > "$results_dir/dqn_without_cache.log"

# Run test with Rainbow DQN Agent
echo "===== Rainbow DQN Agent Evaluation ====="
echo "Running with caching..."
$python_cmd "$test_script" --agent rainbow --episodes 10 > "$results_dir/rainbow_with_cache.log"
echo "Running without caching..."
$python_cmd "$test_script" --agent rainbow --episodes 10 --disable-cache > "$results_dir/rainbow_without_cache.log"

# Run test with Actor-Critic Agent
echo "===== Actor-Critic Agent Evaluation ====="
echo "Running with caching..."
$python_cmd "$test_script" --agent actor-critic --episodes 10 > "$results_dir/actor_critic_with_cache.log"
echo "Running without caching..."
$python_cmd "$test_script" --agent actor-critic --episodes 10 --disable-cache > "$results_dir/actor_critic_without_cache.log"

# Run comprehensive comparison
echo "===== Comprehensive Comparison ====="
$python_cmd "$test_script" --compare-from-logs "$results_dir" > "$results_dir/comparison_results.log"

echo
echo "Evaluation complete! Results are saved in the $results_dir directory"
echo "View the summary report at: $results_dir/comparison_results.log"
echo "End time: $(date)"

# Display quick summary
echo
echo "===== QUICK SUMMARY ====="
if [ -f "$results_dir/comparison_results.log" ]; then
    grep "speedup" "$results_dir/comparison_results.log"
    grep "Cache Hit Ratio" "$results_dir/comparison_results.log"
else
    echo "No summary report available yet"
fi 