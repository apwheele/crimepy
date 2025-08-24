'''
Functions for exact distribution
of day in week (or useful for small
sample Benford tests)
'''

import numpy as np
from math import comb


# composition function generated via Claude
def compositions(n, m):
    """
    Calculate all possible compositions of n items into m bins.
    
    A composition is a way of writing n as a sum of m non-negative integers,
    where order matters and zeros are allowed.
    
    Parameters:
    -----------
    n : int
        Total number of items to distribute
    m : int
        Number of bins
        
    Returns:
    --------
    numpy.ndarray
        2D array where each row represents one composition
        Shape: (num_compositions, m)
    """
    if n == 0:
        return np.zeros((1, m), dtype=int)
    if m == 1:
        return np.array([[n]], dtype=int)
    
    # Total number of compositions is C(n+m-1, m-1)
    total_compositions = comb(n + m - 1, m - 1)
    
    # Pre-allocate result array
    result = np.zeros((total_compositions, m), dtype=int)
    
    # Generate compositions using stars and bars method
    idx = 0
    
    def generate_compositions_recursive(remaining_items, remaining_bins, current_composition):
        nonlocal idx
        
        if remaining_bins == 1:
            # Last bin gets all remaining items
            composition = current_composition + [remaining_items]
            result[idx] = composition
            idx += 1
            return
        
        # Try all possible values for current bin (0 to remaining_items)
        for i in range(remaining_items + 1):
            generate_compositions_recursive(
                remaining_items - i, 
                remaining_bins - 1, 
                current_composition + [i]
            )
    
    generate_compositions_recursive(n, m, [])
    return result

def compositions_vectorized(n, m):
    """
    Vectorized version using itertools-style approach with NumPy.
    More efficient for larger values.
    
    Parameters:
    -----------
    n : int
        Total number of items to distribute
    m : int
        Number of bins
        
    Returns:
    --------
    numpy.ndarray
        2D array where each row represents one composition
    """
    if n == 0:
        return np.zeros((1, m), dtype=int)
    if m == 1:
        return np.array([[n]], dtype=int)
    
    # Use stars and bars method with binary representation
    # We need to place m-1 dividers among n+m-1 positions
    total_compositions = comb(n + m - 1, m - 1)
    
    result = np.zeros((total_compositions, m), dtype=int)
    
    # Generate all combinations of divider positions
    from itertools import combinations
    
    positions = list(range(n + m - 1))
    idx = 0
    
    for dividers in combinations(positions, m - 1):
        # Convert divider positions to composition
        dividers = [-1] + list(dividers) + [n + m - 1]
        composition = [dividers[i+1] - dividers[i] - 1 for i in range(m)]
        result[idx] = composition
        idx += 1
    
    return result

def compositions_pure_numpy(n, m):
    """
    Pure NumPy implementation using meshgrid for small cases.
    Most efficient for small n and m.
    """
    if n == 0:
        return np.zeros((1, m), dtype=int)
    if m == 1:
        return np.array([[n]], dtype=int)
    
    # For larger cases, fall back to recursive method
    if n > 10 or m > 6:
        return compositions(n, m)
    
    # Generate all possible values for each bin
    ranges = [np.arange(n + 1) for _ in range(m)]
    
    # Create meshgrid
    grids = np.meshgrid(*ranges, indexing='ij')
    
    # Stack and reshape to get all combinations
    all_combinations = np.stack(grids, axis=-1).reshape(-1, m)
    
    # Filter to only include valid compositions (sum equals n)
    valid_mask = all_combinations.sum(axis=1) == n
    
    return all_combinations[valid_mask]

# Alias for the most efficient general-purpose function
def composition(n, m):
    """
    Main function to calculate compositions. Automatically chooses
    the most efficient method based on input size.
    """
    if n <= 10 and m <= 6:
        return compositions_pure_numpy(n, m)
    else:
        return compositions(n, m)


