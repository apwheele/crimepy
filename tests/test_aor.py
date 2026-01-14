'''
Test cases for aoristic analysis
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from crimepy.aoristic import (
    weekhour_func_vectorized,
    agg_weekhour,
    agg_hour,
    WEEKDAY_HOUR_INDEX
)

def create_test_data():
    """
    Create test data for aoristic analysis with various time scenarios
    """
    
    # Base date for testing (Monday)
    base_date = datetime(2024, 1, 1, 0, 0, 0)  # Jan 1, 2024 is a Monday
    
    test_cases = []
    
    # Test Case 1: Same hour (01:00 to 01:00) - point in time
    test_cases.append({
        'case': 'same_hour_point',
        'begin': base_date + timedelta(hours=1),
        'end': base_date + timedelta(hours=1),
        'description': '01:00 to 01:00 - point in time'
    })
    
    # Test Case 2: 01:00 to 02:30 - crosses hour boundary 
    test_cases.append({
        'case': 'cross_hour_boundary',
        'begin': base_date + timedelta(hours=1),
        'end': base_date + timedelta(hours=2, minutes=30),
        'description': '01:00 to 02:30 - crosses hour boundary'
    })
    
    # Test Case 3: 01:00 to 01:30 - within same hour
    test_cases.append({
        'case': 'within_same_hour',
        'begin': base_date + timedelta(hours=1),
        'end': base_date + timedelta(hours=1, minutes=30),
        'description': '01:00 to 01:30 - within same hour'
    })
    
    # Test Case 4: 01:50 to 02:10 - crosses hour with partial times
    test_cases.append({
        'case': 'cross_hour_partial',
        'begin': base_date + timedelta(hours=1, minutes=50),
        'end': base_date + timedelta(hours=2, minutes=10),
        'description': '01:50 to 02:10 - crosses hour with partial times'
    })
    
    # Test Case 5: Over 1 day long (25 hours)
    test_cases.append({
        'case': 'over_one_day',
        'begin': base_date + timedelta(hours=10),
        'end': base_date + timedelta(hours=35),  # Next day at 11:00
        'description': 'Over 1 day long (25 hours)'
    })
    
    # Test Case 6: Over 1 week long (8 days)
    test_cases.append({
        'case': 'over_one_week',
        'begin': base_date + timedelta(hours=12),
        'end': base_date + timedelta(days=8, hours=12),  # 8 days later
        'description': 'Over 1 week long (8 days)'
    })
    
    # Test Case 7: Cross midnight boundary
    test_cases.append({
        'case': 'cross_midnight',
        'begin': base_date + timedelta(hours=23, minutes=30),
        'end': base_date + timedelta(days=1, hours=1, minutes=15),
        'description': '23:30 to 01:15 next day - crosses midnight'
    })
    
    # Test Case 8: Cross weekend boundary 
    test_cases.append({
        'case': 'cross_weekend',
        'begin': base_date + timedelta(days=6, hours=22),  # Sunday 22:00
        'end': base_date + timedelta(days=7, hours=2),     # Monday 02:00
        'description': 'Sunday 22:00 to Monday 02:00 - crosses week boundary'
    })
    
    # Test Case 9: Missing end time (should use begin time)
    test_cases.append({
        'case': 'missing_end',
        'begin': base_date + timedelta(hours=15, minutes=30),
        'end': pd.NaT,
        'description': '15:30 with missing end time'
    })
    
    # Test Case 10: Missing begin time (should be skipped)
    test_cases.append({
        'case': 'missing_begin',
        'begin': pd.NaT,
        'end': base_date + timedelta(hours=15, minutes=30),
        'description': 'Missing begin time with 15:30 end'
    })
    
    # Test Case 11: Swapped times (end before begin)
    test_cases.append({
        'case': 'swapped_times',
        'begin': base_date + timedelta(hours=14, minutes=30),
        'end': base_date + timedelta(hours=12, minutes=15),
        'description': 'Swapped times - end before begin'
    })
    
    # Create DataFrame
    df = pd.DataFrame(test_cases)
    
    # Add some grouping variables for testing
    df['crime_type'] = ['Burglary', 'Theft', 'Assault', 'Vandalism', 
                       'Robbery', 'Fraud', 'Burglary', 'Theft',
                       'Assault', 'Missing', 'Vandalism']
    
    df['district'] = ['North', 'South', 'East', 'West', 
                     'North', 'South', 'East', 'West',
                     'North', 'South', 'East']
    
    # Add incident IDs
    df['incident_id'] = range(1, len(df) + 1)
    
    return df

def create_large_test_data(n_rows=10000):
    """
    Create larger dataset for performance testing
    """
    np.random.seed(42)  # For reproducible results
    
    base_date = datetime(2024, 1, 1)
    
    # Random start times over a month
    start_times = [
        base_date + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        ) for _ in range(n_rows)
    ]
    
    # Create end times with various durations
    end_times = []
    for start in start_times:
        duration_type = np.random.choice(['short', 'medium', 'long', 'very_long'], 
                                       p=[0.4, 0.3, 0.2, 0.1])
        
        if duration_type == 'short':  # 0-2 hours
            duration = timedelta(minutes=np.random.randint(0, 120))
        elif duration_type == 'medium':  # 2-8 hours  
            duration = timedelta(hours=np.random.randint(2, 8))
        elif duration_type == 'long':  # 8-24 hours
            duration = timedelta(hours=np.random.randint(8, 24))
        else:  # very_long - over 1 day
            duration = timedelta(days=np.random.randint(1, 14))
            
        end_times.append(start + duration)
    
    # Add some missing data
    missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    for idx in missing_indices:
        if np.random.random() < 0.5:
            end_times[idx] = pd.NaT
        else:
            start_times[idx] = pd.NaT
    
    # Create DataFrame
    df = pd.DataFrame({
        'begin': start_times,
        'end': end_times,
        'crime_type': np.random.choice(['Burglary', 'Theft', 'Assault', 'Vandalism', 'Robbery'], 
                                     size=n_rows),
        'district': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 
                                   size=n_rows),
        'incident_id': range(1, n_rows + 1)
    })
    
    return df

def print_test_results(df, result_df):
    """
    Print human-readable test results
    """
    print("=== AORISTIC ANALYSIS TEST RESULTS ===\n")
    
    for idx, row in df.iterrows():
        print(f"Case {idx+1}: {row['description']}")
        print(f"  Begin: {row['begin']}")
        print(f"  End: {row['end']}")
        
        if isinstance(result_df, list):
            # If we have individual results per row
            weights = result_df[idx]
            non_zero = [(i, w) for i, w in enumerate(weights) if w > 0]
        else:
            # If we have aggregated results, this is more complex to extract
            non_zero = []
        
        if non_zero:
            print(f"  Non-zero weights:")
            for bin_idx, weight in non_zero[:5]:  # Show first 5
                weekday = bin_idx // 24
                hour = bin_idx % 24
                wd_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday]
                print(f"    {wd_name} {hour:02d}:00: {weight:.3f}")
            if len(non_zero) > 5:
                print(f"    ... and {len(non_zero)-5} more")
        else:
            print(f"  No weights (likely missing data)")
        print()

# Create the test datasets
if __name__ == "__main__":
    # Small test dataset
    test_df = create_test_data()
    print("Small test dataset created:")
    print(test_df[['case', 'begin', 'end', 'description']])
    print(f"\nDataset shape: {test_df.shape}")
    
    # Large test dataset
    large_df = create_large_test_data(1000)
    print(f"\nLarge test dataset created with {len(large_df)} rows")
    print(large_df.head())
    
    # Example usage with your original function:
    print("\n=== TESTING WITH ORIGINAL FUNCTIONS ===")
    # Assuming you have the original functions available:
    # result = agg_weekhour(test_df, 'begin', 'end', 'crime_type')
    # print(result.head(10))

# Sample expected results for manual verification:
expected_results = {
    'same_hour_point': "Should have weight 1.0 at Monday 01:00",
    'cross_hour_boundary': "Should split between Monday 01:00 and 02:00",
    'within_same_hour': "Should have weight 1.0 at Monday 01:00", 
    'cross_hour_partial': "Should split between Monday 01:00 and 02:00 based on minutes",
    'over_one_day': "Should distribute evenly across all 168 weekday-hour bins",
    'over_one_week': "Should distribute evenly across all 168 weekday-hour bins"
}

print("\n=== EXPECTED RESULTS FOR VERIFICATION ===")
for case, expected in expected_results.items():
    print(f"{case}: {expected}")


# ============== PYTEST TEST FUNCTIONS ==============

class TestWeekHourFuncVectorized:
    """Tests for the weekhour_func_vectorized function"""

    def test_same_hour_point(self):
        """Test case where begin and end are the same time (point in time)"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=1)],
            'end': [base_date + timedelta(hours=1)]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should have weight 1.0 at Monday hour 1 (index = 0*24 + 1 = 1)
        expected_idx = WEEKDAY_HOUR_INDEX[(0, 1)]  # Monday, 01:00
        assert result[0, expected_idx] == 1.0
        assert result[0].sum() == pytest.approx(1.0)
        assert np.count_nonzero(result[0]) == 1

    def test_within_same_hour(self):
        """Test case where begin and end are within the same hour"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=1)],
            'end': [base_date + timedelta(hours=1, minutes=30)]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should have weight 1.0 at Monday hour 1
        expected_idx = WEEKDAY_HOUR_INDEX[(0, 1)]
        assert result[0, expected_idx] == 1.0
        assert result[0].sum() == pytest.approx(1.0)

    def test_cross_hour_boundary(self):
        """Test case where time spans cross hour boundary"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=1)],
            'end': [base_date + timedelta(hours=2, minutes=30)]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should have weights at Monday hours 1 and 2
        idx_hour1 = WEEKDAY_HOUR_INDEX[(0, 1)]
        idx_hour2 = WEEKDAY_HOUR_INDEX[(0, 2)]

        assert result[0, idx_hour1] > 0
        assert result[0, idx_hour2] > 0
        assert result[0].sum() == pytest.approx(1.0)

    def test_cross_hour_partial(self):
        """Test case where time spans partial hours (01:50 to 02:10)"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=1, minutes=50)],
            'end': [base_date + timedelta(hours=2, minutes=10)]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        idx_hour1 = WEEKDAY_HOUR_INDEX[(0, 1)]
        idx_hour2 = WEEKDAY_HOUR_INDEX[(0, 2)]

        # 10 minutes in hour 1, 10 minutes in hour 2 -> equal weights
        assert result[0, idx_hour1] == pytest.approx(0.5, rel=0.01)
        assert result[0, idx_hour2] == pytest.approx(0.5, rel=0.01)
        assert result[0].sum() == pytest.approx(1.0)

    def test_over_one_week(self):
        """Test case where duration > 1 week gets equal distribution"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=12)],
            'end': [base_date + timedelta(days=8, hours=12)]  # 8 days
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should distribute evenly across all 168 bins
        expected_weight = 1.0 / 168
        assert result[0].sum() == pytest.approx(1.0)
        assert np.allclose(result[0], expected_weight)

    def test_cross_midnight(self):
        """Test case crossing midnight boundary"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=23, minutes=30)],
            'end': [base_date + timedelta(days=1, hours=1, minutes=15)]  # Tuesday 01:15
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should have weights at Monday 23:00, Tuesday 00:00, Tuesday 01:00
        idx_mon_23 = WEEKDAY_HOUR_INDEX[(0, 23)]
        idx_tue_00 = WEEKDAY_HOUR_INDEX[(1, 0)]
        idx_tue_01 = WEEKDAY_HOUR_INDEX[(1, 1)]

        assert result[0, idx_mon_23] > 0
        assert result[0, idx_tue_00] > 0
        assert result[0, idx_tue_01] > 0
        assert result[0].sum() == pytest.approx(1.0)

    def test_missing_end_time(self):
        """Test case where end time is missing (should use begin)"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=15, minutes=30)],
            'end': [pd.NaT]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should use begin time, so weight at Monday 15:00
        expected_idx = WEEKDAY_HOUR_INDEX[(0, 15)]
        assert result[0, expected_idx] == 1.0
        assert result[0].sum() == pytest.approx(1.0)

    def test_missing_begin_time(self):
        """Test case where begin time is missing (should return zeros)"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [pd.NaT],
            'end': [base_date + timedelta(hours=15, minutes=30)]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should have all zeros (invalid row)
        assert result[0].sum() == 0.0

    def test_swapped_times(self):
        """Test case where end is before begin (should auto-swap)"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=14, minutes=30)],
            'end': [base_date + timedelta(hours=12, minutes=15)]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        # Should auto-correct and have weights at hours 12, 13, 14
        idx_hour12 = WEEKDAY_HOUR_INDEX[(0, 12)]
        idx_hour13 = WEEKDAY_HOUR_INDEX[(0, 13)]
        idx_hour14 = WEEKDAY_HOUR_INDEX[(0, 14)]

        assert result[0, idx_hour12] > 0
        assert result[0, idx_hour13] > 0
        assert result[0, idx_hour14] > 0
        assert result[0].sum() == pytest.approx(1.0)

    def test_multiple_rows(self):
        """Test vectorized processing of multiple rows"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [
                base_date + timedelta(hours=1),
                base_date + timedelta(hours=5),
                base_date + timedelta(hours=10)
            ],
            'end': [
                base_date + timedelta(hours=1),
                base_date + timedelta(hours=5),
                base_date + timedelta(hours=10)
            ]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        assert result.shape == (3, 168)
        assert result[0, WEEKDAY_HOUR_INDEX[(0, 1)]] == 1.0
        assert result[1, WEEKDAY_HOUR_INDEX[(0, 5)]] == 1.0
        assert result[2, WEEKDAY_HOUR_INDEX[(0, 10)]] == 1.0


class TestAggWeekHour:
    """Tests for the agg_weekhour function"""

    def test_basic_aggregation(self):
        """Test basic aggregation without grouping"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=1), base_date + timedelta(hours=1)],
            'end': [base_date + timedelta(hours=1), base_date + timedelta(hours=1)]
        })

        result = agg_weekhour(df, 'begin', 'end')

        assert 'weekday' in result.columns
        assert 'hour' in result.columns
        assert 'weight' in result.columns
        assert len(result) == 168
        assert result['weight'].sum() == pytest.approx(2.0)  # Two events

        # Check Monday hour 1 has weight 2.0
        mon_1 = result[(result['weekday'] == 0) & (result['hour'] == 1)]
        assert mon_1['weight'].values[0] == pytest.approx(2.0)

    def test_grouped_aggregation(self):
        """Test aggregation with a grouping variable"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [
                base_date + timedelta(hours=1),
                base_date + timedelta(hours=1),
                base_date + timedelta(hours=2)
            ],
            'end': [
                base_date + timedelta(hours=1),
                base_date + timedelta(hours=1),
                base_date + timedelta(hours=2)
            ],
            'crime_type': ['Burglary', 'Theft', 'Burglary']
        })

        result = agg_weekhour(df, 'begin', 'end', 'crime_type')

        assert 'crime_type' in result.columns
        assert len(result) == 168 * 2  # 168 bins x 2 groups

        # Check Burglary group
        burglary = result[result['crime_type'] == 'Burglary']
        burglary_hour1 = burglary[(burglary['weekday'] == 0) & (burglary['hour'] == 1)]
        burglary_hour2 = burglary[(burglary['weekday'] == 0) & (burglary['hour'] == 2)]
        assert burglary_hour1['weight'].values[0] == pytest.approx(1.0)
        assert burglary_hour2['weight'].values[0] == pytest.approx(1.0)

        # Check Theft group
        theft = result[result['crime_type'] == 'Theft']
        theft_hour1 = theft[(theft['weekday'] == 0) & (theft['hour'] == 1)]
        assert theft_hour1['weight'].values[0] == pytest.approx(1.0)


class TestAggHour:
    """Tests for the agg_hour function"""

    def test_hour_aggregation(self):
        """Test aggregation by hour (collapsing weekdays)"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)  # Monday
        df = pd.DataFrame({
            'begin': [
                base_date + timedelta(hours=1),                    # Monday 01:00
                base_date + timedelta(days=1, hours=1),            # Tuesday 01:00
                base_date + timedelta(hours=2)                     # Monday 02:00
            ],
            'end': [
                base_date + timedelta(hours=1),
                base_date + timedelta(days=1, hours=1),
                base_date + timedelta(hours=2)
            ]
        })

        result = agg_hour(df, 'begin', 'end')

        assert 'hour' in result.columns
        assert 'weight' in result.columns
        assert len(result) == 24

        # Hour 1 should have weight 2.0 (Monday + Tuesday)
        hour1 = result[result['hour'] == 1]
        assert hour1['weight'].values[0] == pytest.approx(2.0)

        # Hour 2 should have weight 1.0
        hour2 = result[result['hour'] == 2]
        assert hour2['weight'].values[0] == pytest.approx(1.0)

    def test_hour_aggregation_with_group(self):
        """Test hour aggregation with grouping"""
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        df = pd.DataFrame({
            'begin': [base_date + timedelta(hours=1), base_date + timedelta(hours=1)],
            'end': [base_date + timedelta(hours=1), base_date + timedelta(hours=1)],
            'district': ['North', 'South']
        })

        result = agg_hour(df, 'begin', 'end', 'district')

        assert 'district' in result.columns
        assert 'hour' in result.columns
        assert len(result) == 24 * 2  # 24 hours x 2 districts


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame({'begin': [], 'end': []})
        df['begin'] = pd.to_datetime(df['begin'])
        df['end'] = pd.to_datetime(df['end'])

        result = agg_weekhour(df, 'begin', 'end')

        assert len(result) == 168
        assert result['weight'].sum() == 0.0

    def test_all_missing_data(self):
        """Test with all missing values"""
        df = pd.DataFrame({
            'begin': [pd.NaT, pd.NaT],
            'end': [pd.NaT, pd.NaT]
        })

        result = weekhour_func_vectorized(df['begin'], df['end'])

        assert result.sum() == 0.0

    def test_weights_sum_to_one_per_row(self):
        """Verify that weights sum to 1.0 for each valid row"""
        test_df = create_test_data()

        result = weekhour_func_vectorized(test_df['begin'], test_df['end'])

        for idx, row in test_df.iterrows():
            if pd.notna(row['begin']):
                # Valid rows should sum to approximately 1.0
                row_sum = result[idx].sum()
                assert row_sum == pytest.approx(1.0, rel=0.01), \
                    f"Row {idx} ({row['case']}): weights sum to {row_sum}, expected 1.0"
            else:
                # Invalid rows should sum to 0.0
                assert result[idx].sum() == 0.0