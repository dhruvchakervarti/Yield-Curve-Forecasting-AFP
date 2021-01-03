#!/usr/bin/env python
# coding: utf-8

# # Yield Curve Forecasting

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from scipy.optimize import minimize
spt= pd.read_csv("Spot_Curve_Monthly.csv")

def nss_model(t,b1,b2,b3,b4,lam1,lam2):
    st= b1+ b2*((1-np.exp(-lam1*t))/(lam1*t))+b3*(((1-np.exp(-lam1*t))/(lam1*t))-np.exp(-lam1*t))+b4*(((1-np.exp(-lam2*t))/(lam2*t))-np.exp(-lam2*t))
    return st


    

def est_param_nss(y,t,lam1,lam2):
 
    on= np.ones(t.shape[0])
    X= np.vstack([on,(1-np.exp(-lam1*t))/(lam1*t),((1-np.exp(-lam1*t))/(lam1*t))-np.exp(-lam1*t),((1-np.exp(-lam2*t))/(lam2*t))-np.exp(-lam2*t)])
    X= np.transpose(X)
    X= np.matrix(X)
    B= np.linalg.inv(X.T*X)*X.T*y
    
    return np.array(B.T)[0]
 


# In[ ]:


#Computing the values of l1 and l2 that define humps, capturing medium and long term term effects

#first hump at 30 months

t=30
lam1= minimize(lambda l:-1*(((1-np.exp(-l*t))/(l*t))-np.exp(-l*t)),x0=[0.6]).x[0]
t=180
lam2= minimize(lambda l:-1*(((1-np.exp(-l*t))/(l*t))-np.exp(-l*t)),x0=[0.6]).x[0]
lam1


# In[ ]:


beta_df= pd.DataFrame({'Date':spt.iloc[:,0].values,'B1':np.zeros(spt.shape[0]),'B2':np.zeros(spt.shape[0]),'B3':np.zeros(spt.shape[0]),'B4':np.zeros(spt.shape[0])})
t= np.array([3,6,12,2*12,3*12,5*12,7*12,10*12,15*12,20*12,25*12,30*12])
for i in range(beta_df.shape[0]):
    y= np.matrix(spt.iloc[i,1:].values).T
    beta_df.iloc[i,1:]= est_param_nss(y,t,lam1,lam2)


# In[ ]:


import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import datetime as dt
beta_changes= pd.read_csv("betas.csv")
beta_df['Date']= pd.to_datetime(beta_df['Date'],format= '%d/%m/%Y')
beta_df= beta_df.set_index(beta_df['Date'])
beta_df= beta_df.iloc[:,1:]
beta_df


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(10,6))

for i, ax in enumerate(axes.flatten()):
    data = beta_df[beta_df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    ax.set_title(beta_df.columns[i])
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout()


# In[ ]:


#Testing Causality

from statsmodels.tsa.stattools import grangercausalitytests
maxlag=4
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
grangers_causation_matrix(beta_df, variables = beta_df.columns)


# In[ ]:


from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.9':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(90%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(beta_df, 0.1)


# ### Looking at the above, a causation may be established:
# 
# B1 can be caused by B2, B3 and B4. B2 can be caused by B1,B3 and B4 B3 can be casued by B1,B2,B4 B4 seems to be caused by B1,B2,B3 Indicatively a VAR model appears to be useful in establishing a relationship with a reasonable significance level.
# 
# Also looking at the Cointegration test, there appears to be a long term relationship between the slope and curvature parameteres with B1, this is line with economic reasoning that B1, the level that is dictated predominatly the short term interest rates seems to be related to backward looking realizations of macroeconomic outcomes(inflation and output gap).

# # Performing test of stationarity

# In[ ]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
 
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# In[ ]:


for name, column in beta_df.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[ ]:


df_train= beta_df.iloc[0:120,:]
df_test= beta_df.iloc[120:,:]
df_differenced = df_train
print(df_train.shape)
print(df_test.shape)


# In[ ]:


for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[ ]:


model = VAR(df_differenced)
for i in [1,2,3,4]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


# In[14]:


# Looking at the AIC above it appears that a lag of 1 may be optimal.


# In[ ]:


model_fitted = model.fit(1)
model_fitted.summary()


# #### Checking serial correlation of residuals The closer they are to a value of 2 the more unlikely is the serial correlation. A value close to zero implies positive serial correlation and a value close to 4 implies a negative serial correlation.

# In[ ]:


from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(beta_df.columns, out):
    print(col, ':', round(val, 2))


# In[ ]:


df_params= model_fitted.params
df_params


# In[ ]:


# Export to Excel 

