'''
The spatial point patter test (sppt)

Andresen's original uses a bootstrapping approach
this uses chi-square with false discovery rate
correction

This relies on a newer version of scipy 1.11
'''


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, false_discovery_control


def chi2(x,correction=True):
    r1 = x.values.astype(float).reshape((2,2)).T
    chi2_contingency(r1,correction=correction)
    s,p,d,e = chi2_contingency(r1,correction=correction)
    return p


def sppt(df,c1,c2,method='by',correction=True,remove_zero=True):
    '''
    spatial point pattern test using Chi-square as false
    discovery rate correction. By default eliminates areas
    with both 0 values
    
    df - pandas dataframe
    c1 - string field with the count for one
    c2 - string field with the count for the other
    method - string, default 'by', could also be 'bh'
             'by' can handle correlated tests
             see false_discovery_control in scipy
    remove_zero - boolean, default True, if both c1 & c2 
                  are 0 removes that area
    
    returns a dataframe with p and q values. P are unadjusted
    and q are the false positive rate adjusted p-values
    
    The index from the original df should flow forward, so could
    be smaller with zeroes removed, but still should be able
    to do typical assignment back to the original dataframe
    '''
    df2 = df[[c1,c2]].copy()
    # removing elements with both zeros
    df2 = df2[df2[[c1,c2]].sum(axis=1) > 0].copy()
    tc1 = df[c1].sum()
    tc2 = df[c2].sum()
    df2['R1'] = tc1 - df2[c1]
    df2['R2'] = tc2 - df2[c2]
    df2['p'] = df2[[c1,c2,'R1','R2']].apply(chi2,correction=correction,axis=1)
    df2['q'] = false_discovery_control(df2['p'],method=method)
    df2['PropC1'] = df2[c1]/tc1
    df2['PropC2'] = df2[c2]/tc2
    df2['Dif'] = df2['PropC2'] - df2['PropC1']
    return df2[[c1,c2,'PropC1','PropC2','Dif','p','q']]


# poisson contours for charts
def pois_contour(df,pre_crime,post_crime,lev=[-3,0,3],
                 lr=5,hr=None,steps=1000):
    '''
    res.plot(color='grey',legend=False,ax=ax)
    '''
    ov_inc = df[post_crime].sum()/df[pre_crime].sum()
    lev = np.array(lev)
    if hr is None:
        hrc = df[pre_crime].max()*1.05
    else:
        hrc = hr
    
    # generate on the square root scale
    gr = np.linspace(np.sqrt(lr),np.sqrt(hrc),steps)**2
    # now just filling in the info I want
    cont_data = np.tile(gr,lev.shape[0])
    lev_data = np.repeat(lev,gr.shape[0])
    df = pd.DataFrame(zip(cont_data,lev_data),columns=['x','levels'])
    df['inc'] = df['x']*ov_inc
    df['vari'] = df['x']*(ov_inc**2)
    df['y'] = df['inc'] + df['levels']*np.sqrt(df['vari'])
    df['y'] = df['y'].clip(0)
    # reshape long to wide
    df = df.pivot(index='x',values='y',columns='levels')
    return df