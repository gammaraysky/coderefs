#### CONFIDENCE INTERVALS AND HYPOTHESIS TESTS ####
############### HANAFI HAFFIDZ ####################
from scipy import stats
import numpy as np
import statsmodels.api as sm
import scipy.stats.distributions as dist
from statsmodels.stats.proportion import proportion_confint

#### ASSUMPTIONS ####
# for one proportion / proportion difference
# - random sample
# - large enough sample size, at least 10 of each outcome

# for one mean
# - random sample
# - large enough sample size for CLT to approx normal, or normal distribution of variable of interest
#   - check with histogram and qqplot

# for mean difference (paired)
# - random sample
# - large enough sample size for CLT to approx normal, or normal distribution of variable of interest
#   - check with histogram and qqplot

# for mean difference (indep)
# - random sample
# - independent
# - large enough sample size for CLT to approx normal, or normal distribution of variable of interest
#   - check with histogram and qqplot
# - checking pooled or unpooled approach
#   - compare sd, boxplot IQRs, if similar, can use pooled (default: unpooled)
#### ASSUMPTIONS ####


# ci_prop(n, sample_prop, alpha=0.05)
# ci_propdiff(n1, n2, sample_prop1, sample_prop2, alpha=0.05)
# ci_mean(data, alpha=0.05)
# ci_meandiffpaired(data1, data2, alpha=0.05)
# ci_meandiffindep(data1, data2, pooled=False, alpha=0.05)

# ht_prop(n, sample_prop, hyp_prop, alt='two-sided', alpha=0.05)
# ht_propdiff(n1, n2, sample_prop1, sample_prop2, hyp_diff=0, alt='two-sided', alpha=0.05)
# ht_mean(data, hyp_mean, alt='two-sided', alpha=0.05)
# ht_meandiffpaired(data1, data2, hyp_diff=0, alt='two-sided', alpha=0.05)
# ht_meandiffindep(data1, data2, hyp_diff=0, alt='two-sided', pooled=False, alpha=0.05)


def ci_prop(n, sample_prop, alpha=0.05):
    # n             - sample size, int
    # sample_prop   - sample proportion, float (0,1)
    # alpha         - significance level. 0.05 for 95% CI
    # RETURNS (lower bound, upper bound)
    p = sample_prop
    se_est = np.sqrt((p*(1-p))/(n))
    z = stats.norm.ppf(1-alpha/2)
    moe = z * se_est
    lcb = sample_prop-moe
    ucb = sample_prop+moe
    print( proportion_confint(sample_prop*n, n, alpha=alpha, method='normal') )
    return lcb, ucb

def ci_propdiff(n1, n2, sample_prop1, sample_prop2, alpha=0.05):
    p1 = sample_prop1
    p2 = sample_prop2
    se1_est = np.sqrt(p1*(1-p1)/n1)
    se2_est = np.sqrt(p2*(1-p2)/n2)
    se_diff_est = np.sqrt(se1_est**2 + se2_est**2)
    z = stats.norm.ppf(1-alpha/2)
    moe = z * se_diff_est
    ucb = (p1 - p2) + moe
    lcb = (p1 - p2) - moe
    return lcb, ucb

def ci_mean(data, alpha=0.05):
    # stats.t.interval(alpha=0.95, loc=np.mean(data), df=len(data)-1, scale=stats.sem(data)) 
    # compared, works accurate

    n = len(data)
    dof = n-1
    sample_mean = np.mean( data )
    sd = np.std( data , ddof=1)
    tstar = stats.t.ppf(1-alpha/2, dof)
    se_est = sd / np.sqrt(n) # se = stats.sem( data ) # both work
    moe = tstar * se_est
    lcb = sample_mean-moe
    ucb = sample_mean+moe
    return lcb, ucb

def ci_meandiffpaired(data1, data2, alpha=0.05):
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    diff_samples = data1 - data2                        
    n = len(data1)  

    diff_mean = np.mean(diff_samples)
    diff_sd = np.std(diff_samples, ddof=1)                       
    
    tstar = stats.t.ppf(q = 1-alpha/2, df = n - 1)   
    moe = tstar * diff_sd/np.sqrt(n)
    lcb = diff_mean - moe
    ucb = diff_mean + moe
    return lcb, ucb

def ci_meandiffindep(data1, data2, pooled=False, alpha=0.05):
    data1 = np.array(data1)
    data2 = np.array(data2)
    n1 = len(data1)
    n2 = len(data2)
    sd1 = np.std(data1) 
    sd2 = np.std(data2)
    sample_mean1 = np.mean(data1)
    sample_mean2 = np.mean(data2)
    sample_mean_diff = sample_mean1 - sample_mean2
    se_est=0
    tstar=0
    dof=0
   
    if pooled==True:
        dof=n1+n2-2
        se_est = np.sqrt( ( (n1-1)*(sd1**2) + (n2-1)*(sd2**2)  ) / dof ) * np.sqrt(1/n1 + 1/n2)
    else:
        # for unpooled, this does not use Welch's t-test, but conservative approach on dof:
        dof=np.min([n1-1,n2-1]) # conservative dof.
        # Welch-Satterthwaite equation to approximate dof 
        dof = ( sd1**2/n1 + sd2**2/n2 )**2 / ( ( sd1**4/(n1**2*(n1-1)) + sd2**4/(n2**2*(n2-1)) )  ) 

        se_est = np.sqrt( (sd1**2)/n1 +(sd2**2)/n2 )

    # # my version, when using unpooled approach, does not use Welch to find dof, 
    # # but uses conservative approach of choosing the smaller n-1 of the two samples.
    # # thus, printing ttest_ind version too for backup.
    # t, pv = stats.ttest_ind( data1, data2, equal_var=pooled, alternative = 'two-sided')
    # # doesn't work. totally diff value

    tstar = stats.t.ppf(q = 1-alpha/2, df = dof)   
    
    moe = tstar * se_est
    lcb = sample_mean_diff - moe
    ucb = sample_mean_diff + moe
    return lcb, ucb

def ht_prop(n, sample_prop, hyp_prop, alt='two-sided', alpha=0.05):
    # n             - sample size, int
    # sample_prop   - sample proportion, float (0,1)
    # hyp_prop      - null value, float (0,1)
    # alt          - {'smaller','larger','two-sided'}
    # alpha         - significance level. 0.05 for 95% CI
    # RETURNS (z_test_stat, p_value, reject_h0)
    p = sample_prop
    se_est = np.sqrt((p*(1-p))/(n))
    ztest_stat = (p - hyp_prop)/se_est

    if (alt=='two-sided'):
        pval = 2*stats.norm.cdf(-abs(ztest_stat))
        # pval = 2*stats.norm.sf(abs(ztest_stat))  # two-tailed p-value
    elif (alt=='smaller' or alt=='less'):
        pval = stats.norm.cdf(ztest_stat)
        # pval = stats.norm.sf(abs(ztest_stat))    # right-tailed p-value
    elif (alt=='larger' or alt=='greater' ):
        pval = stats.norm.cdf(-ztest_stat)
        # pval = stats.norm.sf(-ztest_stat)        # left-tailed p-value
    else:
        print('Error\nUsage: ht_prop(n, sample_prop, hyp_prop, alt=\'two-sided\', alpha=0.05)')
        return False

    # DEBUG
    ztest_stat2, pval2 = sm.stats.proportions_ztest(sample_prop * n, n, hyp_prop, alternative=alt, prop_var=hyp_prop)
    ztest_stat2, pval2 = sm.stats.proportions_ztest(sample_prop * n, n, hyp_prop, alternative=alt, prop_var=None)
    print(ztest_stat2, pval2)

    reject_h0 = pval < alpha
    return ztest_stat, pval, reject_h0

def ht_propdiff(n1, n2, sample_prop1, sample_prop2, hyp_diff=0, alt='two-sided', alpha=0.05):
    # n1, n2         - sample size, int
    # sample_prop1,2 - sample proportion, float (0,1)
    # hyp_prop       - null value, float (0,1)
    # alt            - {'smaller','larger','two-sided'}
    # alpha          - significance level. 0.05 for 95% CI

    # equation will be treated as sample_prop1 - sample_prop2
    # RETURNS (z_test_stat, p_value, reject_h0)

    p1 = sample_prop1
    p2 = sample_prop2

    # combined population proportion estimate, variance, stderr
    phat = (p1*n1 + p2*n2) / (n1 + n2)
    va = phat * (1 - phat)
    se  = np.sqrt(va * (1/n1 + 1/n2)) # as per Umich course
    # TODO check formula, this looks like unpooled:
    se2 = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) # found online, formula seems slightly diff, looks like unpooled 

    # Test statistic and its p-value
    ztest_stat  = ((p1 - p2)- hyp_diff) / se 
    test_stat2 = ((p1 - p2) - hyp_diff)/ se2
    
    #pval = 2*dist.norm.cdf(-np.abs(test_stat))

    if (alt=='two-sided'):
        pval = 2*stats.norm.cdf(-abs(ztest_stat))  # two-tailed p-value
    elif (alt=='smaller' or alt=='less'):
        pval = stats.norm.cdf(ztest_stat)    # right-tailed p-value
    elif (alt=='larger' or alt=='greater' ):
        pval = stats.norm.cdf(-ztest_stat)        # left-tailed p-value
    else:
        print('Error\nUsage: ht_propdiff(n1, n2, sample_prop1, sample_prop2, hyp_diff=0, alt=\'two-sided\', alpha=0.05)')
        return False

    reject_h0 = pval < alpha
    return ztest_stat, pval, reject_h0

def ht_mean(data, hyp_mean, alt='two-sided', alpha=0.05):
    test_stat, p_val = stats.ttest_1samp( data, hyp_mean , alternative=alt) # online method
    print(test_stat, p_val)

    n = len(data)
    dof = n-1
    sample_mean = np.mean( data )
    sd = np.std( data , ddof=1)

    se_est = sd / np.sqrt(n) # se = stats.sem( data ) # both work 
    # se_population = pop_sd / np.sqrt(n) # if pop sd is known, use this

    ttest_stat = (sample_mean - hyp_mean)/se_est

    if (alt=='two-sided'):
        pval = 2*stats.t.cdf(-abs(ttest_stat),dof)
    elif (alt=='smaller' or alt=='less'):
        pval = stats.t.cdf(ttest_stat,dof)  # left
    elif (alt=='larger' or alt=='greater' ):
        pval = stats.t.cdf(-ttest_stat,dof) # right
    else:
        print('Error\nUsage: ht_mean(data, hyp_mean, alt=\'two-sided/smaller/larger\', alpha=0.05)')
        return False

    reject_H0 = pval < alpha
    return ttest_stat, pval, reject_H0

def ht_meandiffpaired(data1, data2, hyp_diff=0, alt='two-sided', alpha=0.05):

    data1 = np.array(data1)
    data2 = np.array(data2)
    tstat, pval = stats.ttest_rel(data1, data2, alternative = alt)
    print(tstat, pval)

    diff_samples = data1 - data2                        
    n = len(data1)
    dof=n-1  

    diff_mean = np.mean(diff_samples)
    diff_sd = np.std(diff_samples, ddof=1)      
    se_est = diff_sd / np.sqrt(n)                 
    
    ttest_stat = (diff_mean - hyp_diff)/se_est

    if (alt=='two-sided'):
        pval = 2*stats.t.cdf(-abs(ttest_stat),dof)
    elif (alt=='smaller' or alt=='less'):
        pval = stats.t.cdf(ttest_stat,dof)  # left
    elif (alt=='larger' or alt=='greater' ):
        pval = stats.t.cdf(-ttest_stat,dof) # right
    else:
        print('Error\nUsage: ht_meandiffpaired(data1, data2, hyp_diff=0, alt=\'two-sided\', alpha=0.05)')
        return False

    reject_H0 = pval < alpha
    return ttest_stat, pval, reject_H0
    
def ht_meandiffindep(data1, data2, hyp_diff=0, alt='two-sided', pooled=False, alpha=0.05):
    data1 = np.array(data1)
    data2 = np.array(data2)
    n1 = len(data1)
    n2 = len(data2)
    sd1 = np.std(data1, ddof=1) 
    sd2 = np.std(data2, ddof=1)
    sample_mean1 = np.mean(data1)
    sample_mean2 = np.mean(data2)
    sample_mean_diff = sample_mean1 - sample_mean2

    se_est=0
    dof=0
    if pooled==True:
        dof=n1+n2-2
        se_est = np.sqrt( ( (n1-1)*(sd1**2) + (n2-1)*(sd2**2)  ) / dof ) * np.sqrt(1/n1 + 1/n2)
    else:
        # for unpooled, this does not use Welch's t-test, but conservative approach on dof:
        dof=np.min([n1-1,n2-1])
        # Welch-Satterthwaite equation to approximate dof 
        dof = ( sd1**2/n1 + sd2**2/n2 )**2 / ( ( sd1**4/(n1**2*(n1-1)) + sd2**4/(n2**2*(n2-1)) )  ) 

        se_est = np.sqrt( (sd1**2)/n1 +(sd2**2)/n2 )
    
    ttest_stat = (sample_mean_diff - hyp_diff)/se_est
    if (alt=='two-sided'):
        pval = 2*stats.t.cdf(-abs(ttest_stat),dof)
        # DEBUG
        t, pv = stats.ttest_ind( data1, data2, equal_var=pooled, alternative = alt)
    elif (alt=='smaller' or alt=='less'):
        pval = stats.t.cdf(ttest_stat,dof)  # left
        # DEBUG
        t, pv = stats.ttest_ind( data1, data2, equal_var=pooled, alternative = 'less')
    elif (alt=='larger' or alt=='greater' ):
        pval = stats.t.cdf(-ttest_stat,dof) # right
        # DEBUG
        t, pv = stats.ttest_ind( data1, data2, equal_var=pooled, alternative = 'greater')
    else:
        print('Error\nUsage: ht_meandiffpaired(data1, data2, hyp_diff=0, alt=\'two-sided\', alpha=0.05)')
        return False

    # print(t,pv)
    # compared, accurate.
    reject_H0 = pval < alpha
    return ttest_stat, pval, reject_H0



# ci_prop               ACCURATE    from statsmodels.stats.proportion import proportion_confint
# ht_prop               ACCURATE    import statsmodels.api as sm, sm.stats.proportions_ztest
# ci_propdiff           no_compare
# ht_propdiff           no_compare
# ci_mean               no_compare
# ht_mean               ACCURATE    from scipy import stats, stats.ttest_1samp
# ci_meandiffpaired     no_compare
# ht_meandiffpaired     ACCURATE    from scipy import stats, stats.ttest_rel
# ci_meandiffindep      no_compare    
# ht_meandiffindep      ACCURATE    from scipy import stats, stats.ttest_ind


# data1 = [10,2,9,24,25,3]
# data2 = [1,2,3,4,5,6]
# data3 = np.random.normal(0,1,10000)
# data4 = np.random.normal(1,1,10000)


# ci_prop(200,0.9,0.02)
# ht_prop(200,0.9,0.7,'larger')
# ci_propdiff(200,100,0.7,0.9)
# ht_propdiff(200,100,0.7,0.9,0,'larger')
# ci_mean(data3)
# ht_mean(data3,.03)
# ci_meandiffpaired(data2,data1)
# ht_meandiffpaired(data2,data1,0)

# ci_meandiffindep(data2,data1,pooled=False)
# ht_meandiffindep(data2,data1,pooled=False,alt='less')
# ht_prop(200,0.9,0.8,'two-sided',0.05)

#ht_propdiff
