'''
Different functions for Weighted Displacement
Difference test (WDD), and various other
Poisson test helpers
'''

import pandas as pd
import math
import numpy as np
from collections import namedtuple
from scipy.stats import poisson, norm
from scipy.special import logsumexp

# wdd test
# Define the named tuple for results
WDDResults = namedtuple('WDDResults', [
    'Est_Local', 'SE_Local', 'Est_Displace', 'SE_Displace', 
    'Est_Total', 'SE_Total', 'Z', 'LowCI', 'HighCI'
])

def wdd(control, treated, disp_control=(0, 0), disp_treated=(0, 0),
        time_weights=(1, 1), area_weights=(1, 1, 1, 1),
        alpha=0.1, silent=True):
    """
    Weighted Difference-in-Differences estimation function
    
    Parameters:
    -----------
    control : tuple/list of length 2
        Control group values [pre, post]
    treated : tuple/list of length 2  
        Treatment group values [pre, post]
    disp_control : tuple/list of length 2, default (0, 0)
        Displacement control values [pre, post]
    disp_treated : tuple/list of length 2, default (0, 0)
        Displacement treatment values [pre, post]
    time_weights : tuple/list of length 2, default (1, 1)
        Weights for [pre, post] periods
    area_weights : tuple/list of length 4, default (1, 1, 1, 1)
        Weights for [control, treated, disp_control, disp_treated] areas
    alpha : float, default 0.1
        Significance level for confidence intervals
    silent : bool, default True
        If True, suppress printed output
    
    Returns:
    --------
    WDDResults : named tuple
        Results containing estimates, standard errors, z-score, and confidence intervals
    """
    
    # Generating the weights
    cpre_w = time_weights[0] * area_weights[0]
    cpost_w = time_weights[1] * area_weights[0]
    tpre_w = time_weights[0] * area_weights[1]
    tpost_w = time_weights[1] * area_weights[1]
    dcpre_w = time_weights[0] * area_weights[2]
    dcpos_w = time_weights[1] * area_weights[2]
    dpre_w = time_weights[0] * area_weights[3]
    dpost_w = time_weights[1] * area_weights[3]
    
    # Generating the stats
    est_local = (treated[1]/tpost_w - treated[0]/tpre_w) - (control[1]/cpost_w - control[0]/cpre_w)
    var_local = (treated[1] * (1/tpost_w)**2 + treated[0] * (1/tpre_w)**2 + 
                 control[1] * (1/cpost_w)**2 + control[0] * (1/cpre_w)**2)
    
    est_disp = (disp_treated[1]/dpost_w - disp_treated[0]/dpre_w) - (disp_control[1]/dcpos_w - disp_control[0]/dcpre_w)
    var_disp = (disp_treated[1] * (1/dpost_w)**2 + disp_treated[0] * (1/dpre_w)**2 + 
                disp_control[1] * (1/dcpos_w)**2 + disp_control[0] * (1/dcpre_w)**2)
    
    tot_est = est_local + est_disp
    var_tot = var_local + var_disp
    
    # Inference stats
    level = norm.ppf(1 - alpha/2)
    z_score = tot_est / math.sqrt(var_tot)
    low_ci = tot_est - level * math.sqrt(var_tot)
    high_ci = tot_est + level * math.sqrt(var_tot)
    
    # Printing results
    if not silent:
        print(f'\n\tThe local WDD estimate is {est_local:.1f} ({math.sqrt(var_local):.1f})')
        print(f'\tThe displacement WDD estimate is {est_disp:.1f} ({math.sqrt(var_disp):.1f})')
        print(f'\tThe total WDD estimate is {tot_est:.1f} ({math.sqrt(var_tot):.1f})')
        print(f'\tThe {100*(1-alpha):.0f}% confidence interval is {low_ci:.1f} to {high_ci:.1f}\n')
    
    # Return results as named tuple
    return WDDResults(
        Est_Local=est_local,
        SE_Local=math.sqrt(var_local),
        Est_Displace=est_disp,
        SE_Displace=math.sqrt(var_disp),
        Est_Total=tot_est,
        SE_Total=math.sqrt(var_tot),
        Z=z_score,
        LowCI=low_ci,
        HighCI=high_ci
    )


# combining multiple harm estimates together
# Define the named tuple for harm results
WDDHarmResults = namedtuple('WDDHarmResults', [
    'HarmEst', 'SE_HarmEst', 'Z', 'LowCI', 'HighCI'
])

def wdd_harm(est, se, weight, alpha=0.1, silent=True):
    """
    Calculate WDD harm estimates with weighted aggregation
    
    Parameters:
    -----------
    est : pandas.Series, numpy.ndarray, or array-like
        Estimates to be weighted
    se : pandas.Series, numpy.ndarray, or array-like
        Standard errors corresponding to estimates
    weight : pandas.Series, numpy.ndarray, or array-like
        Weights to apply to estimates
    alpha : float, default 0.1
        Significance level for confidence intervals
    silent : bool, default True
        If True, suppress printed output
    
    Returns:
    --------
    WDDHarmResults : named tuple
        Results containing harm estimate, standard error, z-score, and confidence intervals
    """
    # Convert inputs to numpy arrays for consistent handling
    est = np.asarray(est)
    se = np.asarray(se)
    weight = np.asarray(weight)
    
    # Harm estimates
    harm_est = est * weight
    harm_var = (se**2) * (weight**2)
    tot_harm = np.sum(harm_est)
    tot_harm_var = np.sum(harm_var)
    tot_harm_se = math.sqrt(tot_harm_var)
    
    # Inference stats
    level = norm.ppf(1 - alpha/2)
    z_score = tot_harm / tot_harm_se
    low_ci = tot_harm - level * tot_harm_se
    high_ci = tot_harm + level * tot_harm_se
    
    # Printing results
    if not silent:
        print(f'\n\tThe total WDD harm estimate is {tot_harm:.1f} ({tot_harm_se:.1f})')
        print(f'\tThe {100*(1-alpha):.0f}% confidence interval is {low_ci:.1f} to {high_ci:.1f}\n')
    
    # Return results as named tuple
    return WDDHarmResults(
        HarmEst=tot_harm,
        SE_HarmEst=tot_harm_se,
        Z=z_score,
        LowCI=low_ci,
        HighCI=high_ci
    )


# example use
#mv = wdd.wdd((133,91),(130,74), silent=True)
#th = wdd.wdd((388,305),(327,202), silent=True)
#bu = wdd.wdd((148,97),(398,190), silent=True)
#rob = wdd.wdd((86,64),(144,92), silent=True)
#ass = wdd.wdd((94,67),(183,96), silent=True)
#tot = pd.DataFrame([mv,th,bu,rob,ass],index=['mv','th','bu','rob','ass'])
#tot['weight'] = [3,2,5,7,10]
#wdd_harm(tot['Est_Total'], tot['SE_Total'],tot['weight'],alpha=0.05)


def cum_wdd(data,treat,cont,alpha=0.1):
    """
    Calculate cumulative differences in treated/control
    over time
    
    Parameters:
    -----------
    data : pandas.Dataframe with the data
    treat: string with the treated field (counts over time)
    cont: string with the control field (counts over time)
    alpha: float between 0 and 1 default 0.1 (for 90% confidence intervals)
    
    Returns:
    --------
    pandas dataframe with additional fields estimate reduction over time with standard errors
    """
    df = data[[treat,cont]].copy()
    # Overall Counts
    df['CumTreat'] = df[treat].cumsum()
    df['CumCont'] = df[cont].cumsum()
    df['CumDif'] = df['CumTreat'] - df['CumCont']
    df['SE'] = np.sqrt(df['CumTreat'] + df['CumCont'])
    z = norm.ppf(1-alpha/2)
    df['LowCI'] = df['CumDif'] - z*df['SE']
    df['HighCI'] = df['CumDif'] + z*df['SE']
    # Normalized per unit time
    norm_punit = np.arange(1,df.shape[0]+1)
    df['CumNorm'] = df['CumDif']/norm_punit
    inv_sq = (1/norm_punit)**2
    df['SENorm'] = np.sqrt(inv_sq*df['CumTreat'] + inv_sq*df['CumCont'])
    df['LowCI_Norm'] = df['CumNorm'] - z*df['SENorm']
    df['HighCI_Norm'] = df['CumNorm'] + z*df['SENorm']
    # IRR statistic (for percentage change) [clipping to avoid errors]
    # presumably should only do this for actual counts in control
    df['LogIRR'] = np.log(df['CumTreat']/df['CumCont'].clip(1))
    df['SElogIRR'] = np.sqrt(1/df['CumTreat'] + 1/df['CumCont'].clip(1))
    df['IRR'] = np.exp(df['LogIRR'])
    df['LowCI_IRR'] = np.exp(df['LogIRR'] - z*df['SElogIRR'])
    df['HighCI_IRR'] = np.exp(df['LogIRR'] + z*df['SElogIRR'])
    df['PercentChangeEst'] = (1 - df['IRR'])*100
    df['LowPercentChange'] = (1 - df['LowCI_IRR'])*100
    df['HighPercentChange'] = (1 - df['HighCI_IRR'])*100
    return df


# E-test functions
def minPMF(r,log_eps=-500):
    """
    Just some simple rules of them to get the minimum logged
    pmf which to calculate the convolution of in the etest function
    """
    if r < 10:
        x = np.arange(0,np.ceil(r)*40+1)
    elif r < 100:
        x = np.arange(0,np.ceil(r)*10+1)
    elif r < 1000:
        x = np.arange(0,np.ceil(r)*3+1)
    else:
        x = np.arange(0,np.ceil(r)*2+1)
    rv = poisson.logpmf(x,mu=r)
    ti = rv>log_eps
    return rv[ti], x[ti]


def etest(k1, k2, n1=1, n2=1, d=0, log_eps=-500):
    """
    Numerically stable version of E-test for large k1, k2 values.
    Uses log-space calculations to avoid overflow/underflow.
    
    Tests the null k1/n1 = k2/n2 + d
    
    k1 - float, poisson mean for value 1
    k2 - float, poisson mean for value 2
    n1 - float, rate for value 1, default 1
    n2 - float, rate for value 2, default 1
    d - adds a constant to value 2 for the null
        so can have a null that rate2 is always 1 more than
        rate 1, etc.
    
    returns a p-value (float)
    """
    if (k1+k2) == 0:
        print('Not defined when both values are 0')
        return None
    if (k1 < 0) | (k2 < 0):
        print('Not defined for negative values')
        return None
    nf1, nf2 = float(n1), float(n2)
    r1, r2 = k1/nf1, k2/nf2
    
    # Check for potential numerical issues
    if r1 > 1000 or r2 > 1000:
        print(f"Warning: Large values detected (rate1={r1}, rate2={r2}). Beware may have numerical issues. Also will take longer to calculate (and may run out of memory).")
        print("May just want to make the denominator bigger to make smaller rates")
    
    lhat = (k1 + k2)/(nf1 + nf2) - (d*nf1)/(nf1 + nf2)
    
    # Avoid division by zero or negative lhat
    if lhat <= 0:
        if d == 0:
            return 1.0  # No evidence against null when lhat <= 0
        else:
            lhat = 1e-10  # Small positive value to avoid numerical issues
    
    Tk = abs((r1 - r2 - d)/math.sqrt(lhat))
    
    d1 = nf2 * lhat
    d2 = nf1 * (lhat + d) #if d != 0 else d1
    
    # instead of worrying about looping to get minPMF values
    # generating the bigger vectors and then just lopping them off
    log_p1, x1 = minPMF(d2,log_eps=log_eps)
    log_p2, x2 = minPMF(d1,log_eps=log_eps)
    # using [1:] as I do not want the 0,0 comparison single row
    exp_p1 = np.tile(log_p1,log_p2.shape[0])[1:]
    exp_p2 = np.repeat(log_p2,log_p1.shape[0])[1:]
    x1t = np.tile(x1,log_p2.shape[0])[1:]
    x2t = np.repeat(x2,log_p1.shape[0])[1:]
    rp = np.sqrt((x1t + x2t)/(nf1 + nf2))
    Tx = np.abs((x1t/nf1) - (x2t/nf2) - d)/rp
    gt = Tx >= Tk
    log_probs = exp_p1[gt] + exp_p2[gt]
    if log_probs.shape[0] == 0:
        return 0.0
    else:
        log_p_total = logsumexp(log_probs)
        return math.exp(log_p_total)

# tests for e-test
#etest(3,0) - 0.0884
#etest(0,3) - 0.0884
#etest(6,2) - 0.1749
#etest(0,0) - not defined None
#etest(3,-2) - not defined
#etest(-2,1) - not defined
#etest(20,20) - should not be over 1, due to finite sum is lower than 1

def scanw(L, k, mu, n):
    """
    Scan statistic approximation for counts in moving window.
    
    Naus scan statistic approximation for Poisson counts in moving window over 
    a particular time period.
    
    Parameters
    ----------
    L : int
        Number of time periods in the window
    k : int
        Window scan time period
    mu : float
        Poisson average per single time period
    n : int
        Observed count
    
    Returns
    -------
    float
        Probability value from the scan statistic approximation
    
    Notes
    -----
    When examining counts of items happening in a specific, discrete set of windows,
    e.g. counts of crime per week, one can use the Poisson PMF to determine the 
    probability of getting an observation over a particular value. For example, if 
    you have a mean of 1 per week, the probability of observing a single week with 
    a count of 6 or more is `poisson.sf(5, 1)` (approximately 0.0006). But if you 
    have monitored a series over 5 years (260 weeks), then the expected number of 
    seeing at least one 6 count in the time period is `poisson.sf(5, 1) * 260`, 
    over 15%.
    
    Now imagine we said "in this particular week span, I observed a count of 6". 
    So it is not in pre-specified week, e.g. Monday through Sunday, but examining 
    over *any* particular moving window. Naus (1982) provides an approximation to 
    correct for this moving window scan. In this example, it ends up being close 
    to 50% is the probability of seeing a moving window of 6 events.
    
    References
    ----------
    Naus, J.I. (1982). Approximations for distributions of scan statistics. 
    Journal of the American Statistical Association, 77, 177-183.
    """
    pn2 = p2(mu, n)
    pn3 = p3(mu, n)
    pnr = pn3 / pn2
    res = pn2 * (pnr ** (L - k))
    return 1 - res

def fns(mu, n, s):
    if n < s:
        return 0
    else:
        res = poisson.pmf(n - s, mu)
        return res

def Fns(mu, n, s):
    """
    Cumulative distribution function helper
    """
    if n < s:
        return 0
    else:
        x = n - s
        res = poisson.cdf(x, mu)
        return res

def p2(mu, n):
    """
    Second order probability calculation
    """
    Fn_2 = Fns(mu, n, 1) ** 2
    p1 = (n - 1) * poisson.pmf(n, mu) * fns(mu, n, 2)
    p2 = (n - 1 - mu) * poisson.pmf(n, mu)
    Fn_3 = Fns(mu, n, 3)
    fin = Fn_2 - p1 - p2 * Fn_3
    return fin

def p3(mu, n):
    """
    Third order probability calculation
    """
    Fn_3 = Fns(mu, n, 1) ** 3
    
    a1_1 = 2 * poisson.pmf(n, mu) * Fns(mu, n, 1)
    a1_2 = (n - 1) * Fns(mu, n, 2) - mu * Fns(mu, n, 3)
    A1 = a1_1 * a1_2
    
    a2_1 = 0.5 * poisson.pmf(n, mu) ** 2
    a2_2 = (n - 1) * (n - 2) * Fns(mu, n, 3) - 2 * (n - 2) * mu * Fns(mu, n, 4)
    a2_3 = mu ** 2 * Fns(mu, n, 5)
    A2 = a2_1 * (a2_2 + a2_3)
    
    A3 = 0
    for i in range(1, n):
        fl3 = fns(mu, 2 * n, i) * Fns(mu, i, 1) ** 2
        A3 += fl3
    
    A4 = 0
    for i in range(1, n - 1):
        fl4_1 = fns(mu, 2 * n, i) * poisson.pmf(i, mu)
        fl4_2 = (i - 1) * Fns(mu, i, 2) - mu * Fns(mu, i, 3)
        A4 += fl4_1 * fl4_2
    
    fin = Fn_3 - A1 + A2 + A3 - A4
    return fin