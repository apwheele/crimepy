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


def gtest(v, p=None):
    """
    Likelihood ratio G test for goodness of fit.

    Calculates the G statistic for testing whether observed counts
    deviate significantly from expected probabilities. This is useful
    for small sample goodness-of-fit testing, such as day-of-week
    analysis or Benford's Law tests.

    Parameters:
    -----------
    v : array-like
        Vector of observed counts in each bin
    p : array-like, optional
        Vector of baseline probabilities. Defaults to equal
        probability across all bins (1/len(v))

    Returns:
    --------
    float
        The G test statistic (likelihood ratio statistic)

    Examples:
    ---------
    >>> # Test if crimes are uniformly distributed across 7 days
    >>> observed = [10, 12, 8, 15, 9, 20, 18]
    >>> g_stat = gtest(observed)

    >>> # Test against specific expected probabilities
    >>> observed = [10, 12, 8, 15, 9, 20, 18]
    >>> expected_p = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2]
    >>> g_stat = gtest(observed, expected_p)

    References:
    -----------
    Translated from ptools R package (Wheeler)
    https://github.com/apwheele/ptools
    """
    v = np.asarray(v, dtype=float)
    if p is None:
        p = np.ones(len(v)) / len(v)
    else:
        p = np.asarray(p, dtype=float)

    e = np.sum(v) * p  # expected values
    # r = log(v/e) where v>0 and e>0, else 0
    mask = (v > 0) & (e > 0)
    r = np.zeros_like(v)
    r[mask] = np.log(v[mask] / e[mask])
    g = 2 * np.sum(v * r)
    return g


def kuiper_test(v, p=None):
    """
    Kuiper's V test statistic for goodness of fit.

    Calculates the Kuiper V statistic for testing whether observed counts
    deviate significantly from expected probabilities. This test is particularly
    useful for circular data (e.g., day-of-week patterns) as it is invariant
    to the starting point of the cycle.

    Parameters:
    -----------
    v : array-like
        Vector of observed counts in each bin
    p : array-like, optional
        Vector of baseline probabilities. Defaults to equal
        probability across all bins (1/len(v))

    Returns:
    --------
    float
        The Kuiper V test statistic

    Examples:
    ---------
    >>> # Test if crimes are uniformly distributed across 7 days
    >>> observed = [3, 1, 1, 0, 0, 1, 1]
    >>> v_stat = kuiper_test(observed)

    >>> # Test against specific expected probabilities
    >>> observed = [3, 1, 1, 0, 0, 1, 1]
    >>> expected_p = [1/7] * 7
    >>> v_stat = kuiper_test(observed, expected_p)

    Notes:
    ------
    The V statistic is calculated as:
        V = (D+ + Dm) * (sqrt(n) + 0.155 + 0.24/sqrt(n))

    where D+ is the maximum positive deviation of the empirical CDF
    from the expected CDF, Dm is the maximum proportion in any bin,
    and n is the total count.

    References:
    -----------
    Wheeler, A. P. (2016). Testing Serial Crime Events for Randomness
    in Day-of-Week Patterns with Small Samples. Journal of Investigative
    Psychology and Offender Profiling, 13(2), 148-165.

    Translated from ptools R package (Wheeler)
    https://github.com/apwheele/ptools
    """
    v = np.asarray(v, dtype=float)
    if p is None:
        p = np.ones(len(v)) / len(v)
    else:
        p = np.asarray(p, dtype=float)

    n = np.sum(v)
    u = np.cumsum(p)       # expected CDF
    s = v / n              # proportions
    e = np.cumsum(s)       # empirical CDF
    Dp = np.max(e - u)     # max positive deviation
    Dm = np.max(s)         # max proportion in any bin
    sq_n = np.sqrt(n)
    V = (Dp + Dm) * (sq_n + 0.155 + 0.24 / sq_n)
    return V
