#!/usr/bin/python
# coding=utf-8
from __future__ import division
from sklearn.metrics import confusion_matrix
import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA

def evaluate_func(df, self = True, level = 0, threshold = 0):
    '''
    #first column is observed values,second column is predicted values
    '''
    df_ = pd.DataFrame()

    r1 = df[df.columns[0]] - df[df.columns[0]].shift(1)
    
    df_['obs'] = r1[abs(r1) >= level]
    
    if self:
        df_['pre'] = df[df.columns[1]] - df[df.columns[1]].shift(1)
    else: 
        df_['pre'] = df[df.columns[1]] - df[df.columns[0]].shift(1)
    if threshold >= 0: 
        df_[df_<=threshold] = -1 #
        df_[df_>threshold] = 1
    else:
        df_[df_>threshold] = 1
        df_[df_<=threshold] = -1 #

    dfnotna = df_.dropna(0)
    cm = confusion_matrix(dfnotna[dfnotna.columns[0]],dfnotna[dfnotna.columns[1]])
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    #Sensitivity(class) = Recall(class) = TruePositiveRate(class) = TP(class) / ( TP(class) + FN(class) ) 
    #Specificity ( mostly used in 2 class problems )= TrueNegativeRate(class)  = TN(class) / ( TN(class) + FP(class) ) 
    SEN = tp/(tp + fn) #Rising early warning rate
    SPE = tn/ (tn + fp) #Drop early warning rate
    PRE = 1- tp/(tp + fp)  #False alarm rate
    ACC = (tp + tn)/(tp + tn + fp + fn) # Accuracy
    #Average deviation(error) rate 
    s = abs(df[df.columns[0]] - df[df.columns[1]])/df[df.columns[0]]
    MAPE = s.mean()
    return MAPE,ACC,SEN,SPE



def test_stationarity(timeseries, window=365):
    #check stady
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window,center=False).mean() 
    rolstd = timeseries.rolling(window=window,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

def trend_split(ts, freq = 52):
    #trand split
    decomposition = seasonal_decompose(ts.values, freq = freq )
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ts[:], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend[:], label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal[:],label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual[:], label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

def cal_wave_pcc(df, threshold_up = '75%',threshold_down = '25%'):
    #df: first columns is observed values,second columns is predicted columns
    from scipy.stats.stats import pearsonr
    df = df.dropna()
    
    if threshold_up == '75%':
        threshold_up = df.describe().loc['75%'][0]
    if threshold_down == '25%':
        threshold_down = df.describe().loc['25%'][0]
    nums = df[df.columns[0]].tolist()
    index = df.index
    peaks = [[0, 0, nums[0],index[0]]]
    oh=[]
    op=[]
    p_values =[]
    dat = []
    len_ = []
    for idx in range(1, len(nums)-1):
        if nums[idx-1] <= nums[idx] > nums[idx+1]:
            peaks.append([1, idx, nums[idx], index[idx]])

        if nums[idx-1] > nums[idx] <= nums[idx+1]:
            peaks.append([0, idx, nums[idx], index[idx]])

    peaks.append([0, len(nums)-1, nums[len(nums)-1], index[len(index)-1]])

    print u'总个数',len(peaks) 
    print u'波谷个数', len([x for x in peaks if x[0] ==0])
    print u'波峰个数', len([x for x in peaks if x[0] ==1])
    peak_interval = []
    for i in range(0,len(peaks)):
        if peaks[i][0] and peaks[i][2]> threshold_up: #peak threshold
            op.append(peaks[i][2])
            dat.append(peaks[i][3])
            F=peaks[i-1][1]
            L=peaks[i+1][1]
            p_pcc = pearsonr(df[df.columns[0]][F:L+1], df[df.columns[1]][F:L+1])
            len_.append(L+1 - F)
            oh.append(p_pcc[0])
            p_values.append(p_pcc[1])
            peak_interval.append(df[df.columns[0]][F:L+1])
    df_re=pd.DataFrame({'pcc_values':oh,'total_nums':len_,'true_values':op,'p_values':p_values,'types':'peak'},index = dat)

    op1 = []
    dat1=[]
    oh1 =[]
    p_values1=[]
    len_1 = []
    vally_interval = []
    for i in range(0,len(peaks)):
        if not peaks[i][0] and peaks[i][2]< threshold_down: #valley threshold
            op1.append(peaks[i][2])
            dat1.append(peaks[i][3])
            F=peaks[i-1][1]
            L=peaks[i+1][1]
            p_pcc1 = pearsonr(df[df.columns[0]][F:L+1], df[df.columns[1]][F:L+1])
            vally_interval.append(df[df.columns[0]][F:L+1])
            len_1.append(L+1 - F)
            oh1.append(p_pcc1[0])
            p_values1.append(p_pcc1[1])

    df_re1=pd.DataFrame({'pcc_values':oh1,'total_nums':len_1,'true_values':op1,'p_values':p_values1,'types':'valley'},index = dat1)    
    try:
        ts2 = pd.concat(vally_interval)
        ts1 = pd.concat(peak_interval)
        ts = pd.concat([ts1, ts2]).to_frame(name = 'coverage')
    except:ts = []
    df_f = pd.concat([df_re,df_re1])

    #calculate average
    #ss.groupby('types')['p_values','pcc_values'].mean()
    ss = df_f.dropna()
    ave  = ss.groupby('types')['p_values','pcc_values'].mean()
    
    fig, ax = plt.subplots(dpi = 200)
    ax.set(title='SZKB predicted vs. observed values')
    #ax.plot(df.join(ts)['coverage'], 'pink', label='coverage',lw = 10,alpha = 1)
    ax.plot(df[df.columns[0]], 'orange', label='observed')
    ax.plot(df[df.columns[1]], 'g', label='predicted')

    ax.plot(df.join(ss[ss.types == 'peak'])['true_values'].fillna(0), 'r',label='peak')
    ax.plot(df.join(ss[ss.types == 'valley'])['true_values'].fillna(0), 'b',label='vally')

    for x in ss.index:
        if ss.loc[[x]]['pcc_values'].values.round(3)[0] < 0.5:
            ax.text(x,ss.loc[[x]]['true_values'].values,ss.loc[[x]]['pcc_values'].values.round(3)[0],size = 14)
    plt.legend(fontsize = 14)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Number',fontsize = 20)
    return ss,ts,ave