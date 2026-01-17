'''
Test cases for exact distribution functions
including compositions and G-test
'''

import numpy as np
from math import comb
import pytest

from crimepy.exact import (
    compositions,
    compositions_vectorized,
    compositions_pure_numpy,
    composition,
    gtest,
    kuiper_test
)


class TestCompositions:
    """Tests for the compositions function (recursive implementation)"""

    def test_basic_composition(self):
        """Test basic composition of 3 items into 2 bins"""
        result = compositions(3, 2)

        # Should have C(3+2-1, 2-1) = C(4,1) = 4 compositions
        assert result.shape == (4, 2)

        # All rows should sum to 3
        assert np.all(result.sum(axis=1) == 3)

        # Expected compositions: [0,3], [1,2], [2,1], [3,0]
        expected = {(0, 3), (1, 2), (2, 1), (3, 0)}
        actual = {tuple(row) for row in result}
        assert actual == expected

    def test_zero_items(self):
        """Test composition of 0 items"""
        result = compositions(0, 3)

        assert result.shape == (1, 3)
        assert np.all(result == 0)

    def test_single_bin(self):
        """Test composition into single bin"""
        result = compositions(5, 1)

        assert result.shape == (1, 1)
        assert result[0, 0] == 5

    def test_count_formula(self):
        """Verify composition count matches stars-and-bars formula"""
        for n in range(1, 8):
            for m in range(1, 5):
                result = compositions(n, m)
                expected_count = comb(n + m - 1, m - 1)
                assert result.shape[0] == expected_count, \
                    f"compositions({n}, {m}) returned {result.shape[0]} rows, expected {expected_count}"

    def test_all_sums_correct(self):
        """Verify all compositions sum to n"""
        result = compositions(5, 3)
        assert np.all(result.sum(axis=1) == 5)

    def test_no_negative_values(self):
        """Verify no negative values in compositions"""
        result = compositions(4, 4)
        assert np.all(result >= 0)


class TestCompositionsVectorized:
    """Tests for the compositions_vectorized function"""

    def test_matches_recursive(self):
        """Verify vectorized matches recursive implementation"""
        for n in range(1, 6):
            for m in range(1, 4):
                result_rec = compositions(n, m)
                result_vec = compositions_vectorized(n, m)

                # Same shape
                assert result_rec.shape == result_vec.shape

                # Same set of compositions (order may differ)
                set_rec = {tuple(row) for row in result_rec}
                set_vec = {tuple(row) for row in result_vec}
                assert set_rec == set_vec, \
                    f"Mismatch for compositions({n}, {m})"

    def test_zero_items(self):
        """Test composition of 0 items"""
        result = compositions_vectorized(0, 3)
        assert result.shape == (1, 3)
        assert np.all(result == 0)

    def test_single_bin(self):
        """Test composition into single bin"""
        result = compositions_vectorized(5, 1)
        assert result.shape == (1, 1)
        assert result[0, 0] == 5


class TestCompositionsPureNumpy:
    """Tests for the compositions_pure_numpy function"""

    def test_matches_recursive_small(self):
        """Verify pure numpy matches recursive for small inputs"""
        for n in range(1, 8):
            for m in range(1, 5):
                result_rec = compositions(n, m)
                result_np = compositions_pure_numpy(n, m)

                # Same set of compositions
                set_rec = {tuple(row) for row in result_rec}
                set_np = {tuple(row) for row in result_np}
                assert set_rec == set_np, \
                    f"Mismatch for compositions_pure_numpy({n}, {m})"

    def test_fallback_for_large_inputs(self):
        """Test that large inputs fall back to recursive method"""
        # n > 10 should trigger fallback
        result = compositions_pure_numpy(12, 3)
        expected_count = comb(12 + 3 - 1, 3 - 1)
        assert result.shape[0] == expected_count


class TestCompositionAlias:
    """Tests for the composition function (main entry point)"""

    def test_small_inputs_use_numpy(self):
        """Verify small inputs produce correct results"""
        result = composition(5, 3)
        expected_count = comb(5 + 3 - 1, 3 - 1)
        assert result.shape[0] == expected_count

    def test_large_inputs(self):
        """Verify large inputs produce correct results"""
        result = composition(15, 3)
        expected_count = comb(15 + 3 - 1, 3 - 1)
        assert result.shape[0] == expected_count
        assert np.all(result.sum(axis=1) == 15)

    def test_edge_cases(self):
        """Test edge cases"""
        # Zero items
        result = composition(0, 5)
        assert result.shape == (1, 5)
        assert np.all(result == 0)

        # Single bin
        result = composition(10, 1)
        assert result.shape == (1, 1)
        assert result[0, 0] == 10


class TestGTest:
    """Tests for the G-test (likelihood ratio test) function"""

    def test_uniform_null(self):
        """Test G statistic against uniform null hypothesis"""
        observed = [10, 12, 8, 15, 9, 20, 18]
        g_stat = gtest(observed)

        # Should return a positive value
        assert g_stat > 0
        # Known value from manual calculation
        assert g_stat == pytest.approx(9.6731, rel=0.01)

    def test_custom_probabilities(self):
        """Test G statistic with custom expected probabilities"""
        observed = [10, 12, 8, 15, 9, 20, 18]
        expected_p = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2]
        g_stat = gtest(observed, expected_p)

        assert g_stat > 0
        assert g_stat == pytest.approx(1.8277, rel=0.01)

    def test_perfect_fit(self):
        """Test that perfect fit returns G = 0"""
        # Equal counts with uniform null -> G should be 0
        perfect = [10, 10, 10, 10]
        g_stat = gtest(perfect)

        assert g_stat == pytest.approx(0.0, abs=1e-10)

    def test_perfect_fit_custom_probs(self):
        """Test perfect fit with custom probabilities"""
        # Counts proportional to probabilities -> G should be 0
        observed = [10, 20, 30, 40]  # Total = 100
        expected_p = [0.1, 0.2, 0.3, 0.4]
        g_stat = gtest(observed, expected_p)

        assert g_stat == pytest.approx(0.0, abs=1e-10)

    def test_with_zeros(self):
        """Test handling of zero counts"""
        observed_zeros = [0, 10, 5, 0, 8]
        g_stat = gtest(observed_zeros)

        # Should handle zeros without error
        assert g_stat > 0
        assert g_stat == pytest.approx(25.2186, rel=0.01)

    def test_all_zeros_except_one(self):
        """Test extreme case with all counts in one bin"""
        observed = [100, 0, 0, 0]
        g_stat = gtest(observed)

        # Very high G value expected
        assert g_stat > 0
        # G = 2 * 100 * ln(100 / 25) = 200 * ln(4) = 277.26
        assert g_stat == pytest.approx(277.26, rel=0.01)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input"""
        observed = np.array([10, 12, 8, 15])
        expected_p = np.array([0.25, 0.25, 0.25, 0.25])

        g_stat = gtest(observed, expected_p)
        assert g_stat > 0

    def test_list_input(self):
        """Test that lists work as input"""
        observed = [10, 12, 8, 15]
        g_stat = gtest(observed)
        assert g_stat > 0

    def test_default_probabilities(self):
        """Test that default probabilities are uniform"""
        observed = [10, 20, 30]

        # Explicit uniform probabilities
        g_explicit = gtest(observed, [1/3, 1/3, 1/3])

        # Default (should be same)
        g_default = gtest(observed)

        assert g_explicit == pytest.approx(g_default, rel=1e-10)

    def test_two_bins(self):
        """Test simple two-bin case"""
        observed = [60, 40]
        g_stat = gtest(observed)

        # Manual calculation: G = 2 * (60*ln(60/50) + 40*ln(40/50))
        # G = 2 * (60*ln(1.2) + 40*ln(0.8))
        # G = 2 * (60*0.1823 + 40*(-0.2231))
        # G = 2 * (10.94 - 8.92) = 4.04
        assert g_stat == pytest.approx(4.02, rel=0.02)

    def test_benford_law_example(self):
        """Test with Benford's law expected distribution"""
        # First digit distribution according to Benford's law
        benford_p = [
            np.log10(1 + 1/d) for d in range(1, 10)
        ]

        # Sample observed data (hypothetical)
        observed = [301, 176, 125, 97, 79, 67, 58, 51, 46]

        g_stat = gtest(observed, benford_p)

        # Should return a valid G statistic
        assert g_stat >= 0
        assert np.isfinite(g_stat)

    def test_day_of_week_example(self):
        """Test typical day-of-week crime analysis"""
        # Crimes by day of week (Mon-Sun)
        observed = [45, 42, 38, 51, 67, 89, 68]

        g_stat = gtest(observed)

        # Weekend effect should show high G value
        assert g_stat > 0
        assert np.isfinite(g_stat)


class TestGTestEdgeCases:
    """Edge case tests for gtest function"""

    def test_single_bin(self):
        """Test with single bin (degenerate case)"""
        observed = [100]
        g_stat = gtest(observed)

        # Single bin with p=1.0 -> G = 0
        assert g_stat == pytest.approx(0.0, abs=1e-10)

    def test_very_small_counts(self):
        """Test with very small counts"""
        observed = [1, 1, 1, 1]
        g_stat = gtest(observed)

        assert g_stat == pytest.approx(0.0, abs=1e-10)

    def test_large_counts(self):
        """Test with large counts"""
        observed = [10000, 12000, 8000, 15000]
        g_stat = gtest(observed)

        # Should still work correctly
        assert g_stat > 0
        assert np.isfinite(g_stat)

    def test_probability_sum_not_one(self):
        """Test behavior when probabilities don't sum to 1"""
        observed = [10, 20, 30]
        # Probabilities that don't sum to 1
        prob = [0.2, 0.2, 0.2]  # sum = 0.6

        # Function should still compute (uses probabilities as given)
        g_stat = gtest(observed, prob)
        assert np.isfinite(g_stat)

    def test_float_counts(self):
        """Test that float counts work (e.g., weighted counts)"""
        observed = [10.5, 12.3, 8.7, 15.2]
        g_stat = gtest(observed)

        assert g_stat > 0
        assert np.isfinite(g_stat)


class TestCompositionsIntegration:
    """Integration tests combining composition and gtest functions"""

    def test_exact_test_workflow(self):
        """Test typical workflow for exact small sample test"""
        # Generate all possible ways 10 events could fall into 4 bins
        n = 10
        m = 4
        all_compositions = composition(n, m)

        # Verify we got the expected number
        expected_count = comb(n + m - 1, m - 1)
        assert len(all_compositions) == expected_count

        # Calculate G statistic for each composition
        g_values = []
        for comp in all_compositions:
            if comp.sum() > 0:  # Skip if all zeros
                g = gtest(comp)
                g_values.append(g)

        # All G values should be non-negative
        assert all(g >= 0 for g in g_values)

        # Perfect uniform distribution should have G = 0
        # For n=10, m=4: uniform is not achievable with integers
        # but [2,2,3,3] or similar should be close to 0
        close_to_uniform = np.array([2, 3, 2, 3])
        g_near_uniform = gtest(close_to_uniform)
        assert g_near_uniform < 1.0  # Should be small

    def test_observed_vs_distribution(self):
        """Test comparing observed data against exact distribution"""
        observed = np.array([5, 3, 2, 0])
        n = observed.sum()  # 10
        m = len(observed)   # 4

        # Calculate observed G statistic
        g_observed = gtest(observed)

        # Generate all compositions
        all_comps = composition(n, m)

        # Calculate G for all possible outcomes
        g_all = np.array([gtest(comp) for comp in all_comps])

        # Count how many have G >= observed (for p-value calculation)
        count_extreme = np.sum(g_all >= g_observed)

        # This count should be reasonable (not all or none)
        assert 0 < count_extreme < len(g_all)


class TestKuiperTest:
    """Tests for the Kuiper V test function"""

    def test_uniform_null(self):
        """Test Kuiper V statistic against uniform null hypothesis"""
        observed = [3, 1, 1, 0, 0, 1, 1]
        v_stat = kuiper_test(observed)

        # Should return a positive value
        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_custom_probabilities(self):
        """Test Kuiper V with custom expected probabilities"""
        observed = [3, 1, 1, 0, 0, 1, 1]
        expected_p = [1/7] * 7
        v_stat = kuiper_test(observed, expected_p)

        # Should match default uniform probabilities
        v_default = kuiper_test(observed)
        assert v_stat == pytest.approx(v_default, rel=1e-10)

    def test_perfect_fit(self):
        """Test that perfect uniform fit returns low V"""
        # Equal counts with uniform null -> V should be small
        perfect = [10, 10, 10, 10]
        v_stat = kuiper_test(perfect)

        # With perfect fit, empirical CDF matches expected CDF
        # Dp = 0, but Dm = 0.25 (max proportion), so V > 0
        assert v_stat > 0
        assert v_stat < 2.0  # Should be relatively small

    def test_extreme_concentration(self):
        """Test with all counts in one bin"""
        observed = [100, 0, 0, 0]
        v_stat = kuiper_test(observed)

        # Very high V value expected due to extreme deviation
        assert v_stat > 5.0

    def test_with_zeros(self):
        """Test handling of zero counts"""
        observed_zeros = [0, 10, 5, 0, 8]
        v_stat = kuiper_test(observed_zeros)

        # Should handle zeros without error
        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input"""
        observed = np.array([10, 12, 8, 15])
        expected_p = np.array([0.25, 0.25, 0.25, 0.25])

        v_stat = kuiper_test(observed, expected_p)
        assert v_stat > 0

    def test_list_input(self):
        """Test that lists work as input"""
        observed = [10, 12, 8, 15]
        v_stat = kuiper_test(observed)
        assert v_stat > 0

    def test_day_of_week_example(self):
        """Test typical day-of-week crime analysis"""
        # Crimes by day of week (Mon-Sun) from ptools example
        observed = [3, 1, 1, 0, 0, 1, 1]
        v_stat = kuiper_test(observed)

        # Known value from R ptools package
        # V = 2.065331 (approximately)
        assert v_stat == pytest.approx(2.0653, rel=0.01)

    def test_larger_sample(self):
        """Test with larger sample size"""
        observed = [10, 12, 8, 15, 9, 20, 18]
        v_stat = kuiper_test(observed)

        # Known value from testing
        assert v_stat == pytest.approx(2.1243, rel=0.01)

    def test_two_bins(self):
        """Test simple two-bin case"""
        observed = [60, 40]
        v_stat = kuiper_test(observed)

        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_single_bin(self):
        """Test with single bin (degenerate case)"""
        observed = [100]
        v_stat = kuiper_test(observed)

        # Single bin: s = [1.0], e = [1.0], u = [1.0]
        # Dp = max(e - u) = 0, Dm = max(s) = 1.0
        # V = (0 + 1) * (10 + 0.155 + 0.024) = 10.179
        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_formula_verification(self):
        """Verify the V statistic formula manually"""
        observed = [4, 2, 2, 2]  # n=10
        p = [0.25, 0.25, 0.25, 0.25]

        v_stat = kuiper_test(observed, p)

        # Manual calculation
        n = 10
        s = np.array([0.4, 0.2, 0.2, 0.2])
        e = np.cumsum(s)  # [0.4, 0.6, 0.8, 1.0]
        u = np.cumsum(p)  # [0.25, 0.5, 0.75, 1.0]
        Dp = np.max(e - u)  # max([0.15, 0.1, 0.05, 0.0]) = 0.15
        Dm = np.max(s)  # 0.4
        sq_n = np.sqrt(n)
        expected_V = (Dp + Dm) * (sq_n + 0.155 + 0.24 / sq_n)

        assert v_stat == pytest.approx(expected_V, rel=1e-10)


class TestKuiperTestEdgeCases:
    """Edge case tests for kuiper_test function"""

    def test_very_small_counts(self):
        """Test with very small total count"""
        observed = [1, 1, 1, 1]
        v_stat = kuiper_test(observed)

        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_large_counts(self):
        """Test with large counts"""
        observed = [10000, 12000, 8000, 15000]
        v_stat = kuiper_test(observed)

        # Should still work correctly
        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_many_bins(self):
        """Test with many bins"""
        observed = [5, 3, 7, 2, 8, 4, 6, 1, 9, 5]
        v_stat = kuiper_test(observed)

        assert v_stat > 0
        assert np.isfinite(v_stat)

    def test_float_counts(self):
        """Test that float counts work (e.g., weighted counts)"""
        observed = [10.5, 12.3, 8.7, 15.2]
        v_stat = kuiper_test(observed)

        assert v_stat > 0
        assert np.isfinite(v_stat)
