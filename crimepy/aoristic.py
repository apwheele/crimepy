'''
Functions to conduct aoristic
analysis using python

Claude rewrite to make more efficient
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants - precompute for efficiency
WEEK_HOUR_TUPLES = [(wd, hr) for wd in range(7) for hr in range(24)]
WEEK_HOUR_DF = pd.DataFrame(WEEK_HOUR_TUPLES, columns=['weekday', 'hour'])
WEEK_EQUAL = np.full(168, 1.0/168)  # Use numpy array for speed
WD_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
WD_DICT = {i: w for i, w in enumerate(WD_LABELS)}
WEEK_HOUR_LABELS = [f'{wd}_{hr}' for wd in WD_LABELS for hr in range(24)]

# Precompute weekday-hour index mapping for O(1) lookups
WEEKDAY_HOUR_INDEX = {(wd, hr): idx for idx, (wd, hr) in enumerate(WEEK_HOUR_TUPLES)}

# Constants needed
week_hour = [(wd,hr) for wd in range(7) for hr in range(24)]
week_hour_df = pd.DataFrame(week_hour,columns=['weekday','hour'])
week_equal = [1.0/len(week_hour)]*len(week_hour)
week_missing = [None]*len(week_hour)
wd_lab = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
wd_di = {i:w for i,w in enumerate(wd_lab)}
week_hour_labs = [f'{wd}_{hr}' for wd in wd_lab for hr in range(24)]
week_color = ['#BABABA', '#878787', '#3F0001', '#7F0103', '#D6604D', '#F4A582', '#FDDBC7']

# Claude rewrite to make more efficient
def weekhour_func_vectorized(begin_series, end_series):
    '''
    Vectorized version of weekhour_func that processes entire series at once
    
    Parameters:
    -----------
    begin_series : pd.Series of datetime
    end_series : pd.Series of datetime
    
    Returns:
    --------
    np.ndarray of shape (n_rows, 168) with weights for each weekday-hour combination
    '''
    n_rows = len(begin_series)
    result = np.zeros((n_rows, 168))
    
    # Handle missing data vectorized
    begin_valid = begin_series.notna()
    end_valid = end_series.notna()
    
    # Where we have begin but no end, use begin as end
    end_filled = end_series.where(end_valid, begin_series)
    
    # Only process rows with valid begin times
    valid_mask = begin_valid
    if not valid_mask.any():
        return result
    
    # Work with valid data only
    begin_valid_data = begin_series[valid_mask]
    end_valid_data = end_filled[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # Handle swapped times
    swapped = begin_valid_data > end_valid_data
    begin_corrected = np.where(swapped, end_valid_data, begin_valid_data)
    end_corrected = np.where(swapped, begin_valid_data, end_valid_data)
    
    # Floor to hour
    begin_floor = pd.Series(begin_corrected).dt.floor('h')
    end_floor = pd.Series(end_corrected).dt.floor('h')
    
    # Calculate duration in minutes
    duration_minutes = (end_corrected - begin_corrected) / pd.Timedelta(minutes=1)
    
    # For very long durations (>1 week), use equal distribution
    long_duration_mask = duration_minutes > 10080
    if long_duration_mask.any():
        long_indices = valid_indices[long_duration_mask]
        result[long_indices] = WEEK_EQUAL
    
    # Process normal durations
    normal_mask = ~long_duration_mask
    if normal_mask.any():
        normal_indices = valid_indices[normal_mask]
        
        begin_normal = begin_corrected[normal_mask]
        end_normal = end_corrected[normal_mask]
        begin_floor_normal = begin_floor[normal_mask]
        end_floor_normal = end_floor[normal_mask]
        duration_normal = duration_minutes[normal_mask]
        
        # Vectorized processing for each case
        for i, (idx, b_orig, e_orig, b_floor, e_floor, dur_min) in enumerate(
            zip(normal_indices, begin_normal, end_normal, begin_floor_normal, end_floor_normal, duration_normal)
        ):
            if pd.isna(b_orig) or pd.isna(e_orig):
                continue
                
            # Single hour case
            if b_floor == e_floor:
                wd = b_floor.weekday()
                hr = b_floor.hour
                result[idx, WEEKDAY_HOUR_INDEX[(wd, hr)]] = 1.0
                continue
            
            # Multi-hour case - calculate proportions
            total_minutes = (e_orig - b_orig) / pd.Timedelta(minutes=1)
            if total_minutes <= 0:
                continue
                
            # Generate hour range
            hour_range = pd.date_range(b_floor, e_floor, freq='1h')
            
            # Calculate weights for each hour
            weights = []
            for j, hour_dt in enumerate(hour_range):
                if j == 0:  # First hour
                    minutes_in_hour = 60 - (b_orig - b_floor) / pd.Timedelta(minutes=1)
                elif j == len(hour_range) - 1:  # Last hour
                    minutes_in_hour = (e_orig - e_floor) / pd.Timedelta(minutes=1)
                else:  # Middle hours
                    minutes_in_hour = 60
                
                weight = minutes_in_hour / total_minutes
                weights.append((hour_dt.weekday(), hour_dt.hour, weight))
            
            # Aggregate weights by weekday-hour
            for wd, hr, weight in weights:
                if (wd, hr) in WEEKDAY_HOUR_INDEX:
                    result[idx, WEEKDAY_HOUR_INDEX[(wd, hr)]] += weight
    
    return result

def agg_weekhour(data, begin, end, group=None):
    '''
    Optimized version of agg_weekhour using vectorized operations
    
    Parameters:
    -----------
    data : pd.DataFrame
    begin : str, column name for begin datetime
    end : str, column name for end datetime  
    group : str or list, optional grouping column(s)
    
    Returns:
    --------
    pd.DataFrame with aggregated weights by weekday-hour
    '''
    # Get weight matrix (n_rows x 168)
    weights_matrix = weekhour_func_vectorized(data[begin], data[end])
    
    if group is None:
        # Simple case: sum across all rows
        total_weights = weights_matrix.sum(axis=0)
        result_df = WEEK_HOUR_DF.copy()
        result_df['weight'] = total_weights
        return result_df
    
    # Group case: need to aggregate by group
    if isinstance(group, str):
        group = [group]
    
    # Create DataFrame with weights and group columns
    group_data = data[group].copy()
    
    # Handle case where we have grouping variables
    unique_groups = group_data.drop_duplicates().reset_index(drop=True)
    results = []
    
    for _, group_vals in unique_groups.iterrows():
        # Find matching rows
        mask = np.ones(len(data), dtype=bool)
        for col in group:
            mask &= (data[col] == group_vals[col])
        
        if mask.any():
            # Sum weights for this group
            group_weights = weights_matrix[mask].sum(axis=0)
            
            # Create result DataFrame for this group
            group_df = WEEK_HOUR_DF.copy()
            for col in group:
                group_df[col] = group_vals[col]
            group_df['weight'] = group_weights
            
            results.append(group_df)
    
    if results:
        final_result = pd.concat(results, ignore_index=True)
        final_result = final_result[group + ['weekday', 'hour', 'weight']]
        return final_result.sort_values(group + ['weekday', 'hour']).reset_index(drop=True)
    else:
        # No valid groups, return empty result with correct structure
        result_df = WEEK_HOUR_DF.copy()
        for col in group:
            result_df[col] = None
        result_df['weight'] = 0.0
        return result_df[group + ['weekday', 'hour', 'weight']]

# Alternative even faster version for simple cases
def weekhour_func_simple(begin_series, end_series):
    '''
    Simplified version for when you don't need all the edge case handling
    Much faster for clean data
    '''
    n_rows = len(begin_series)
    result = np.zeros((n_rows, 168))
    
    # Basic validation
    valid = begin_series.notna() & end_series.notna()
    if not valid.any():
        return result
    
    begin_clean = begin_series[valid]
    end_clean = end_series[valid]
    valid_idx = np.where(valid)[0]
    
    # Vectorized weekday/hour extraction
    begin_weekday = begin_clean.dt.weekday.values
    begin_hour = begin_clean.dt.hour.values
    end_weekday = end_clean.dt.weekday.values  
    end_hour = end_clean.dt.hour.values
    
    # Calculate duration in hours
    duration_hours = (end_clean - begin_clean).dt.total_seconds() / 3600
    
    # Simple case: same weekday and hour
    same_hour_mask = (begin_weekday == end_weekday) & (begin_hour == end_hour)
    if same_hour_mask.any():
        same_hour_indices = valid_idx[same_hour_mask]
        for i, (idx, wd, hr) in enumerate(zip(same_hour_indices, 
                                            begin_weekday[same_hour_mask], 
                                            begin_hour[same_hour_mask])):
            result[idx, wd * 24 + hr] = 1.0
    
    # Multi-hour cases would need more complex logic...
    # For now, fall back to original method for complex cases
    complex_mask = ~same_hour_mask
    if complex_mask.any():
        # Use original logic for complex cases
        complex_indices = valid_idx[complex_mask]
        for i in complex_indices:
            # Fall back to single-row processing for complex cases
            row_result = weekhour_func_single_row(begin_series.iloc[i], end_series.iloc[i])
            if row_result is not None:
                result[i] = row_result
    
    return result

def weekhour_func_single_row(begin_val, end_val):
    '''Helper function for single row processing'''
    if pd.isna(begin_val):
        return None
    
    if pd.isna(end_val):
        end_val = begin_val
    
    if begin_val > end_val:
        begin_val, end_val = end_val, begin_val
    
    beg = begin_val.floor('h')
    end = end_val.floor('h') 
    
    tot_min = (end_val - begin_val).total_seconds() / 60
    if tot_min > 10080:
        return WEEK_EQUAL.copy()
    
    rv = pd.date_range(beg, end, freq='1h')
    if len(rv) == 1:
        wd, hr = rv[0].weekday(), rv[0].hour()
        result = np.zeros(168)
        result[WEEKDAY_HOUR_INDEX[(wd, hr)]] = 1.0
        return result
    
    # Calculate proportional weights
    beg_sub = (3600 - (begin_val - beg).total_seconds()) / 60
    end_sub = (end_val - end).total_seconds() / 60  
    mid_sub = tot_min - beg_sub - end_sub
    
    weights = [beg_sub] + [mid_sub] * (len(rv) - 2) + [end_sub]
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    result = np.zeros(168)
    for i, (dt, weight) in enumerate(zip(rv, weights)):
        wd, hr = dt.weekday(), dt.hour()
        result[WEEKDAY_HOUR_INDEX[(wd, hr)]] += weight
    
    return result


def agg_hour(data,begin,end,group=None):
    res_week = agg_weekhour(data,begin,end,group)
    if group:
        res_day = res_week.groupby(group + ['hour'],as_index=False)['weight'].sum()
    else:
        res_day = res_week.groupby('hour',as_index=False)['weight'].sum()
    return res_day


def plt_super(data,
              color=week_color,
              ax=None,
              figsize=(8,4),
              show=True,
              legend=True,
              leg_kwargs={'bbox_to_anchor':(1.0,1.0)},
              save=None,
              save_kwarges={'dpi':500,'bbox_inches':'tight'},
              title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for w in range(7):
        sub = data[data['weekday'] == w].copy()
        ax.plot(sub['hour'],sub['weight'],c=week_color[w],label=wd_di[w])
    ax.set_xticks(list(range(0,24,2)))
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(**leg_kwargs)
    if save:
        fig.savefig(save,**save_kwargs)
    if show:
        fig.show()


def plt_basic(data,
              color='k',
              ax=None,
              figsize=(8,4),
              show=True,
              label=None,
              save=None,
              save_kwargs={'dpi':500,'bbox_inches':'tight'},
              title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['hour'],data['weight'],marker='o',c=color,markeredgecolor='white',label=label)
    ax.set_xticks(list(range(0,24,2)))
    if title:
        ax.set_title(title)
    if save:
        fig.savefig(save,**save_kwargs)
    if show:
        fig.show()