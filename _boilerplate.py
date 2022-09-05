############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
import re
import warnings
warnings.filterwarnings('ignore')

sns.set()
sns.set_style("darkgrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# print(plt.rcParams)
plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=10) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the legend
plt.rc('figure', figsize=(8,4))
# plt.style.available
plt.style.use('seaborn-white')
############################################################



##? SCALERS
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
pwrtfm = PowerTransformer()
stdard = StandardScaler()
robust = RobustScaler()
minmax = MinMaxScaler()



##? CLIPPING NORMALIZED FEATURES
def outlierclip(df, max):  
    for col in df.columns:
        df[col] = df[col].apply(lambda x:max if x>max else x)
        df[col] = df[col].apply(lambda x:-max if x<-max else x)
    return df

df_pwrtfm = outlierclip(df_pwrtfm, 3)




##? PLOT ALL HISTOGRAMS AND COUNTPLOTS FOR ALL COLUMNS
fig, axs = plt.subplots(5, 6, figsize=(20,10), constrained_layout=True)
for col, ax in zip(df.columns, axs.ravel()):
    if df[col].dtype=='float64':
        ax.hist(df[col], bins=40)
    else:
        sns.countplot(x=df[col], ax=ax)
    ax.set_title(col)
plt.show()


fig, axs = plt.subplots(2,7, figsize=(15,6), constrained_layout=True)
for col,ax in zip(df_pwrtfm.columns,axs.ravel()):
    ax.boxplot(df_pwrtfm[col])
    ax.set_title(col)
    
plt.show()



##? SHAPIRO-WILKS HYPO TEST FOR NORMALITY
from scipy.stats import shapiro
from termcolor import colored

# Univariate normality test
for col in df.columns:
    stat, p_value = shapiro(df[col])
    alpha = 0.05    # significance level
    if p_value > alpha: 
        result = colored('Accepted', 'green')
    else:
        result = colored('Rejected','red')        
    print('Feature: {}\t p-value: {:.4f}\t Hypothesis: {}'.format(col, p_value, result))



##? POISSON DISPERSION TEST / CHI SQUARED GOODNESS OF FIT?
from scipy.stats import chi2
intcols = [c for c in df.columns if df[c].dtype=='int64']

# Univariate poisson test
for col in df[intcols].columns:
    # Parameters
    alpha = 0.05      
    n = len(df[col])  
    dof = n-1         
    
    # Statistics
    mean = df[col].mean()               # sample mean
    D = ((df[col]-mean)**2).sum()/mean    # test statistic
    
    # Two-tailed test
    q_lower = alpha/2
    q_upper = (1-alpha)/2
    
    # percentile point function = inverse of cdf
    chi2_crit_lower = chi2.ppf(q_lower, dof)
    chi2_crit_upper = chi2.ppf(q_upper, dof)
    
    if (D<chi2_crit_lower) or (D>chi2_crit_upper):
        result = colored('Rejected', 'red')
    else:
        result = colored('Accepted', 'green')
    print('Feature: {}\t Hypothesis: {}'.format(col, result))






### IMPUTE THE REST WITH KNNIMPUTER
from sklearn.impute import KNNImputer

# select numeric columns for imputation
numericcols = [c for c in train.columns if train[c].dtype=='int64'] + [c for c in train.columns if train[c].dtype=='float64']
numericcols = [c for c in numericcols if c!=target]
both = pd.concat([train,test])

# fit and transform
imputer = KNNImputer(n_neighbors=9, weights='distance').fit(train[numericcols])

train[numericcols] = imputer.transform(train[numericcols])
test[numericcols] =  imputer.transform(test[numericcols])