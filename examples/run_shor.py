"""
Example: Running Shor's Factoring Algorithm

This script demonstrates how to use Shor's algorithm to factor integers.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from q_sim.algorithms import shors_algorithm


def main():
    print("=" * 60)
    print("Shor's Factoring Algorithm Demo")
    print("=" * 60)
    
    # Test cases
    test_numbers = [15, 21, 35]
    
    for N in test_numbers:
        print(f"\n{'=' * 60}")
        print(f"Factoring N = {N}")
        print('=' * 60)
        
        # Run Shor's algorithm
        result = shors_algorithm(N)
        
        if result["success"]:
            factors = result["factors"]
            print(f"\n✓ Success!")
            print(f"  Factors: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}")
            print(f"  Method: {result.get('method', 'quantum')}")
            
            if "period" in result:
                print(f"  Period found: {result['period']}")
        else:
            print(f"\n✗ Failed to factor {N}")
            print(f"  Reason: {result.get('message', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("Note: This is a simplified simulation.")
    print("Full quantum period finding requires additional gate types.")
    print("=" * 60)


if __name__ == "__main__":
    main()
