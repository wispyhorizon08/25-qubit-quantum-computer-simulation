"""
Example: Running Grover's Search Algorithm

This script demonstrates how to use Grover's algorithm to search
for a specific state in a quantum superposition.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from q_sim.algorithms import grovers_search


def main():
    print("=" * 60)
    print("Grover's Search Algorithm Demo")
    print("=" * 60)
    
    # Search for '10110' in a 5-qubit system
    num_qubits = 5
    target = "10110"
    
    print(f"\nSearching for target state: {target}")
    print(f"Search space size: 2^{num_qubits} = {2**num_qubits} states")
    
    # Run Grover's algorithm
    print("\nRunning Grover's algorithm...")
    counts = grovers_search(num_qubits, target)
    
    # Display results
    print("\nMeasurement results (top 10):")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (state, count) in enumerate(sorted_counts[:10]):
        probability = count / sum(counts.values()) * 100
        marker = " ← TARGET" if state == target else ""
        print(f"  {state}: {count:4d} ({probability:5.1f}%){marker}")
    
    # Calculate success probability
    target_count = counts.get(target, 0)
    success_prob = target_count / sum(counts.values()) * 100
    
    print(f"\nSuccess probability: {success_prob:.1f}%")
    print(f"Classical random search would have: {100/2**num_qubits:.1f}% success rate")
    print(f"Quantum speedup factor: {success_prob / (100/2**num_qubits):.1f}x")


if __name__ == "__main__":
    main()
