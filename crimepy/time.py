'''
Functions to help create
time series charts
'''

import pandas as pd
import numpy as np
from datetime import timedelta
from copy import copy
import matplotlib.pyplot as plt
from .cdcplot import combo_legend

def weekly_data(data,
                date_field,
                begin_week=None,
                end_week=None,
                smooth=8,
                z=3):
    '''
    Function to calculate weekly error bars. Begin/end taken from data
    assume that there are no partial day reporting
    
    data - pandas dataframe
    date_field - string date field in dataframe
    begin_week - either a specific datestring to start the index,
                 or none determines via data (starting on Monday)
    end_week - either a specific datestring to end the weeks on, or none
               determines via data
    smooth - how many weeks to generate smooth mean estimate,
             default 8
    z - zscore range, default 3
    '''
    d2 = data[~data[date_field].isna()]
    date_min = data[date_field].min().date()
    if begin_week is None:
        begin_week = date_min + timedelta(7 - date_min.weekday())
    else:
        begin_week = pd.to_datetime(f'{date_min.year}-{date_min.month+1}-01')
    if end_week is None:
        # This assumes no partial reporting per day
        end_week = data[date_field].max().date() + timedelta(1)
        week_df = pd.date_range(begin_week,end_week,freq="7D",inclusive='both')[:-1]
    else:
        end_week = pd.to_datetime(end_week) + timedelta(1)
        week_df = pd.date_range(begin_week,end_week,freq="7D",inclusive='both')[:-1]
    week_df = pd.DataFrame(week_df,columns=['Week'])
    npc = np.floor((d2[date_field] - pd.to_datetime(begin_week)).dt.days/7).astype(int).value_counts()
    week_df['Counts'] = npc
    week_df['Counts'] = week_df['Counts'].fillna(0).astype(int)
    week_df['PriorMean'] = week_df['Counts'].rolling(smooth,closed='left').mean()
    week_df['Low'] = ((-z/2 + np.sqrt(week_df['PriorMean'])).clip(0)**2)
    week_df['High'] = (z/2 + np.sqrt(week_df['PriorMean']))**2
    return week_df


def monthly_data(data,
                 date_field,
                 begin=None,
                 end=None):
    '''
    Function to calculate monthly aggregation. Begin/end taken from data
    assume that there are no partial day reporting
    
    data - pandas dataframe
    date_field - string date field in dataframe
    begin - either a specific datestring to specify the month,
            or none determines via data (full month needed from 1st)
    end - either a specific datestring to end months on,
          or taken from data
    '''
    d2 = data[~data[date_field].isna()]
    date_min = data[date_field].min().date()
    if begin is None:
        if date_min.day == 1:
            begin_date = date_min
        else:
            begin_date = pd.to_datetime(f'{date_min.year}-{date_min.month+1}-01')
    else:
        begin = pd.to_datetime(begin)
    if end is None:
        # This assumes no partial reporting per day
        end_date = data[date_field].max().date() + timedelta(1)
        month_df = pd.date_range(begin_date,end_date,freq=pd.offsets.MonthBegin(1),inclusive='both')[:-1]
    else:
        end_date = pd.to_datetime(end) + timedelta(1)
        month_df = pd.date_range(begin_date,end_date,freq=pd.offsets.MonthBegin(1),inclusive='both')[:-1]
    month_df = pd.DataFrame(month_df,columns=['Month'])
    # aggregate to months
    d2 = d2[[date_field]].copy()
    d2['Month'] = ((d2[date_field] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
                         .dt.floor('d'))
    vc_month = d2['Month'].value_counts()
    month_df.set_index('Month',inplace=True)
    month_df['Counts'] = vc_month
    month_df['Counts'] = month_df['Counts'].fillna(0).astype(int)
    return month_df.reset_index()


def month_chart(data,ax=None,file=None,
                line_kwargs={'color':'k',
                             'marker':'o',
                             'markeredgecolor':'w',
                             'markersize':None},
                figsize=(10,5),
                title=None,
                dpi=500,
                annotate=None,
                markersize=None):
    ax_orig = copy(ax)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['Month'],data['Counts'],**line_kwargs)
    if title is None:
        pass
    elif title == '':
        pass
    else:
        ax.set_title(title,loc='left')
    if annotate is None:
        pass
    elif annotate == '':
        pass
    else:
        ax.annotate(annotate, xy=(-0.04, -0.13),
            xycoords='axes fraction', textcoords='offset points',
            size=10, ha='left', va='bottom')
    if file is None:
        plt.show()
    elif file == 'return':
        if ax_orig is None:
            return fig, ax
        else:
            return ax
    else:
        plt.savefig(file,dpi=dpi, bbox_inches='tight')
        plt.clf()


def week_chart(data,ax=None,file=None,max_weeks=52*3,
               figsize=(12,4),
               title=None,
               legend_loc=(0.005, 0.02),legend_kwargs={'prop':{'size':11}},dpi=500,annotate=None):
    # get rid of missing data
    md = data[~data['PriorMean'].isna()].copy()
    if max_weeks > -1:
        md = md.tail(max_weeks)
    ax_orig = copy(ax)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(md['Week'], md['Low'], md['High'],
                    alpha=0.2, color='k', label='Prior 8 Weeks')
    ax.plot(md['Week'], md['PriorMean'], color='k', label='Prior 8 Weeks')
    ax.plot(md['Week'], md['Counts'], color="#286090", linewidth=1.5, label='Actual',
            marker='o', markersize=2)
    ax.set_ylabel(None)
    # Making a nicer legend
    handler, labeler = ax.get_legend_handles_labels()
    hd = [(handler[0],handler[1]),handler[2]]
    ax.legend(hd, [labeler[0],labeler[2]], loc=legend_loc,**legend_kwargs)
    if title is None:
        pass
    elif title == '':
        pass
    else:
        ax.set_title(title,loc='left')
    if annotate is None:
        pass
    elif annotate == '':
        pass
    else:
        ax.annotate(annotate, xy=(-0.04, -0.13),
            xycoords='axes fraction', textcoords='offset points',
            size=10, ha='left', va='bottom')
    if file is None:
        plt.show()
    elif file == 'return':
        if ax_orig is None:
            return fig, ax
        else:
            return ax
    else:
        plt.savefig(file,dpi=dpi, bbox_inches='tight')
        plt.clf()

def group_consecutive_years(years):
    """
    Groups consecutive years into ranges.
    
    Args:
        years: List of integers representing years
    
    Returns:
        List of strings representing year ranges
    """
    if not years:
        return []
    
    # Sort the years to handle unsorted input
    sorted_years = sorted(years)
    ranges = []
    start = sorted_years[0]
    end = sorted_years[0]
    
    for i in range(1, len(sorted_years)):
        if sorted_years[i] == end + 1:
            # Consecutive year, extend the range
            end = sorted_years[i]
        else:
            # Gap found, finalize current range
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = sorted_years[i]
            end = sorted_years[i]
    
    # Add the final range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    return ranges

def seas_chart(data,ax=None,file=None,
               figsize=(10,5),
               leg_kwargs={},year_colors={},title=None,dpi=500,annotate=None):
    md = data.copy()
    md['Year'] = md['Month'].dt.year
    md['MonthN'] = md['Month'].dt.month
    year_list = list(pd.unique(md['Year']))
    year_list.sort()
    last_year = max(year_list)
    lab_hist = f'{min(year_list)}-{max(year_list)-1}'
    if year_colors:
        loyl = list(set(year_list) - set(year_colors.keys()))
        loyl.sort()
        loyl = group_consecutive_years(loyl)
        #loyl = [str(i) for i in loyl]
        lab_hist = ",".join(loyl)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax_orig = copy(ax)
    else:
        ax_orig = copy(ax)
    labN = 0
    for y in year_list:
        subd = md[md['Year'] == y]
        if y < last_year:
            if labN == 0 and y not in year_colors:
                ax.plot(subd['MonthN'],subd['Counts'],color='grey',linewidth=0.8,
                        label=lab_hist)
                labN == 1
            elif y in year_colors:
                ax.plot(subd['MonthN'],subd['Counts'],color=year_colors[y],linewidth=0.8,
                        label=y)
            else:
                ax.plot(subd['MonthN'],subd['Counts'],color='grey',linewidth=0.8)
        else:
            ax.plot(subd['MonthN'],subd['Counts'],color='orange',linewidth=2.1,
                    label=last_year)
            ax.plot(subd['MonthN'].tail(1),subd['Counts'].tail(1),
                    color='orange',marker='o',markeredgecolor='white',
                    markersize=12,label=last_year)
    ax.set_xticks(range(1,13))
    hd, lab = combo_legend(ax,sort=True) # may need to reorder
    ax.legend(hd, lab,**leg_kwargs)
    if title is None:
        pass
    elif title == '':
        pass
    else:
        ax.set_title(title,loc='left')
    if annotate is None:
        pass
    elif annotate == '':
        pass
    else:
        ax.annotate(annotate, xy=(-0.04, -0.13),
            xycoords='axes fraction', textcoords='offset points',
            size=10, ha='left', va='bottom')
    if file is None:
        plt.show()
    elif file == 'return':
        if ax_orig is None:
            return fig, ax
        else:
            return ax
    else:
        plt.savefig(file,dpi=dpi, bbox_inches='tight')
        plt.clf()