"""
Utility functions for quantum algorithms.
"""

import math
from typing import List


def binary_to_int(binary_str: str) -> int:
    """Convert binary string to integer."""
    return int(binary_str, 2)


def int_to_binary(num: int, num_bits: int) -> str:
    """Convert integer to binary string with specified number of bits."""
    return format(num, f'0{num_bits}b')


def reverse_bits(num: int, num_bits: int) -> int:
    """Reverse the bits of a number."""
    binary_str = int_to_binary(num, num_bits)
    reversed_str = binary_str[::-1]
    return binary_to_int(reversed_str)


def gcd(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Compute the least common multiple."""
    return abs(a * b) // gcd(a, b)


def mod_exp(base: int, exponent: int, modulus: int) -> int:
    """
    Compute (base^exponent) mod modulus efficiently.
    
    Uses the square-and-multiply algorithm.
    """
    result = 1
    base = base % modulus
    
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    
    return result


def find_period(f_values: List[int]) -> int:
    """
    Find the period of a periodic function from its values.
    
    Args:
        f_values: List of function values
        
    Returns:
        The period, or 0 if no period found
    """
    n = len(f_values)
    
    for period in range(1, n // 2 + 1):
        is_period = True
        for i in range(period, n):
            if f_values[i] != f_values[i % period]:
                is_period = False
                break
        if is_period:
            return period
    
    return 0


def continued_fraction(numerator: int, denominator: int, max_terms: int = 10) -> List[int]:
    """
    Compute the continued fraction representation of numerator/denominator.
    
    Returns:
        List of continued fraction coefficients
    """
    cf = []
    
    for _ in range(max_terms):
        if denominator == 0:
            break
        
        quotient = numerator // denominator
        cf.append(quotient)
        
        numerator, denominator = denominator, numerator - quotient * denominator
    
    return cf


def convergents(cf: List[int]) -> List[tuple]:
    """
    Compute the convergents of a continued fraction.
    
    Returns:
        List of (numerator, denominator) tuples
    """
    if not cf:
        return []
    
    convergents_list = []
    
    # First convergent
    h_prev2, h_prev1 = 1, cf[0]
    k_prev2, k_prev1 = 0, 1
    convergents_list.append((h_prev1, k_prev1))
    
    # Subsequent convergents
    for i in range(1, len(cf)):
        h_curr = cf[i] * h_prev1 + h_prev2
        k_curr = cf[i] * k_prev1 + k_prev2
        convergents_list.append((h_curr, k_curr))
        
        h_prev2, h_prev1 = h_prev1, h_curr
        k_prev2, k_prev1 = k_prev1, k_curr
    
    return convergents_list
