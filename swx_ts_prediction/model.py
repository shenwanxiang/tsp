# coding=utf-8
#statsmodel's verion should be overthan 0.7
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn import metrics
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def grid_search_ARMA_para(ts, pdqrange = (0,2)):
    import warnings
    import itertools
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(pdqrange[0], pdqrange[1])

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    para_re = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                para_re.append([param, param_seasonal, results.aic, results.bic])
            except:
                continue
    return sorted(para_re, key=lambda x: x[2])[0]

def ARMA_train_predict_rolling(ts_log, roll_begin_index = 5, para_search_step = 20):
    '''
    输入为取了log的ts，输出为实际值和预测值的df（这时候不再是取了log的值）
    lag = 0 表示用前一周的预测后一周的，lag = 1表示用前前1周的预测下一周的
    roll_begin_index：表示滑动预测起始的地方
    VIP：roll_begin_index必须大于lag
    para_search_step表示参数网格搜索的步长，如果为1则表示每增加一个训练集样本参数调整一次（这样做计算量大，所以选取20）
    '''
    import warnings
    lag =0
    test = ts_log.iloc[roll_begin_index:,]
    ts_push = ts_log.shift(lag)

    train = ts_log.iloc[:roll_begin_index,] #第一条数
    
    test_push = ts_push.iloc[roll_begin_index:,]

    history = [x for x in train.dropna()]
    predictions = list()
    aic = list()
    bic = list()
    print('Printing Predicted vs Expected Values...')
    print('\n')
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    change_flag = para_search_step  # 滚动para_search_step步参数网格搜索一次
    best_para = grid_search_ARMA_para(history, pdqrange = (0,2)) # 调用格点搜索函数寻找最佳参数
    for t in range(len(test)):
        change_flag = change_flag-1
        if change_flag <= 0:
            change_flag = para_search_step
            best_para = grid_search_ARMA_para(history, pdqrange = (0,2)) #调用格点搜索函数寻找最佳参数
        model = sm.tsa.statespace.SARIMAX(history,
                                          order=best_para[0], #使用最佳参数
                                          enforce_invertibility = False,
                                          enforce_stationarity=False,
                                          seasonal_order = best_para[1])  #使用最佳参数
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(float(yhat))
        obs = test_push[t]
        history.append(obs)
        aic.append(best_para[2])
        bic.append(best_para[3])
        print('predicted=%f, expected=%f，best_para = %s' % (np.exp(yhat), np.exp(obs), best_para)) #打印结果和最佳参数

    #error = mean_squared_error(np.exp(test), np.exp(predictions))
    predictions_series = pd.Series(predictions, index = test.index)
    aic_series = pd.Series(aic, index = test.index)
    bic_series = pd.Series(bic, index = test.index)
    df_aic_bic = pd.concat([aic_series.to_frame(name = 'AIC'), bic_series.to_frame(name = 'BIC')],axis =1)
    df_log_obs_pre = pd.concat([test.to_frame(name = 'observed'), predictions_series.to_frame(name = 'predicted')],axis =1)
    df_obs_pre = np.exp(df_log_obs_pre)
    return df_obs_pre,df_aic_bic #返回实际值和预测值的DataFrame

def grid_search_ARMA_Ex_para(ts,ex_train, pdqrange = (0,2)):
    import warnings
    import itertools
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(pdqrange[0], pdqrange[1])

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    #s = [4,12,24,26,28]
    s = [12]
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q,s))]

    para_re = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                exog = ex_train,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                para_re.append([param, param_seasonal, results.aic, results.bic])
            except:
                continue
    return sorted(para_re, key=lambda x: x[2])[0]


def ARMA_Ex_train_predict_rolling(ts_log, ex_vectors, roll_begin_index = 51, para_search_step = 20, pdqrange = (0,2)):
    '''
    input: ts_log（ts）and ex_vectors（df）
    output: observed vs. predicted dataframe
	ex_vectors: ex features
    roll_begin_index：rolling begin
    para_search_step: best parameters update step
    '''
    import warnings
    test = ts_log.iloc[roll_begin_index:,]
    train = ts_log.iloc[:roll_begin_index,] 

    ex_test = ex_vectors.iloc[roll_begin_index:,]
    ex_train = ex_vectors.iloc[:roll_begin_index,]

    predictions = list()
    aic = list()
    bic = list()
    print('Printing Predicted vs Expected Values...')
    print('\n')
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    change_flag = para_search_step   # 
    best_para = grid_search_ARMA_Ex_para(train, ex_train, pdqrange = pdqrange) #gridsearch for best parameters
    #model_fit = model.fit(disp=2)
    #output = model_fit.forecast(exog = ex_train.iloc[[-1]])
    for t in range(len(test)):
        change_flag = change_flag-1
        if change_flag == 0:
            change_flag = para_search_step
            print 're-search best para, please wait ...'
            best_para = grid_search_ARMA_Ex_para(train, ex_train, pdqrange = pdqrange) #gridsearch for best parameters

        model = sm.tsa.statespace.SARIMAX(train,
                                          exog = ex_train,
                                          order=best_para[0],
                                          #mle_regression = False,
                                          enforce_invertibility = False,
                                          enforce_stationarity=False,
                                          seasonal_order = best_para[1])
        model_fit = model.fit(disp=2)
        output = model_fit.forecast(exog = ex_test.iloc[[t]])
        yhat = output[0]
        predictions.append(float(yhat))
        obs = test[t]

        train = pd.concat([train,test.iloc[[t]]])
        ex_train = pd.concat([ex_train,ex_test.iloc[[t]]])

        aic.append(best_para[2])
        bic.append(best_para[3])
        print 'predicted = ',round(yhat,3), ' expected = ', round(obs,3),' best_para = ',best_para[0],best_para[1],round(best_para[3],1) #打印结果和最佳参数

    #error = mean_squared_error(np.exp(test), np.exp(predictions))
    predictions_series = pd.Series(predictions, index = test.index)
    aic_series = pd.Series(aic, index = test.index)
    bic_series = pd.Series(bic, index = test.index)
    df_aic_bic = pd.concat([aic_series.to_frame(name = 'AIC'), bic_series.to_frame(name = 'BIC')],axis =1)
    df_log_obs_pre = pd.concat([test.to_frame(name = 'observed'), predictions_series.to_frame(name = 'predicted')],axis =1)
    return df_log_obs_pre,df_aic_bic 

	
def Adaboost_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5):
    #dff first column is y, other columns is x1,x2,x3...
    
    train = dff.iloc[:roll_begin_index,] #第一条数
    test = dff.iloc[roll_begin_index:,]

    #自定义打分函数
    def my_score_(ground_truth, predictions):
        predict1 = np.array(predictions) + 1 

        tr = np.array(ground_truth)
        pre = np.array(predict1)
        error = (np.abs(tr - pre)/pre).mean()
        return error

    my_score = make_scorer(my_score_, greater_is_better=False) #自定义打分函数

    best_scores = []
    output = []
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor())

    clf = grid_search.GridSearchCV(estimator= regr_2,
                             param_grid={
                                         'learning_rate': [0.01, 0.1, 1],
                                         'n_estimators': [300,500,800],
                                        # 'max_depth': [3, 5, 7, 9]
                                        },
                             cv = cv,
                             scoring= cv_score) #scoring see: http://scikit-learn.org/stable/modules/model_evaluation.html

    clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values)
    classifier = clf.best_estimator_
    change_flag = para_search_step
    for i in range(len(test)): 
        change_flag = change_flag-1
        if change_flag == 0:
            change_flag = para_search_step
            print 're-search best para, please wait ...'
            clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values) #x,y
            classifier = clf.best_estimator_
        
        best_scores.append(clf.best_score_)    
        output.append(classifier.predict(test.iloc[[i]][test.iloc[[i]].columns[1:]].values))
        
        train = pd.concat([train,test.iloc[[i]]]) #更新train
        
        print 'Training Set sizes:', len(train), 'score: ', clf.best_score_
        
    pre = pd.DataFrame(output,columns=['predicted'])
    pre.index = test.index
    df_re = pd.concat([test[[test.columns[0]]],pre],axis =1)
    df_re = df_re.rename(columns = {test.columns[0]:'observed'})
    df_re.index = pd.to_datetime(df_re.index)

    scores_re = pd.DataFrame(best_scores,columns=[cv_score])
    scores_re.index = test.index
    return df_re, scores_re

def GBDT_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5):
    #dff first column is y, other columns is x1,x2,x3...
    
    train = dff.iloc[:roll_begin_index,] #第一条数
    test = dff.iloc[roll_begin_index:,]

    #自定义打分函数
    def my_score_(ground_truth, predictions):
        predict1 = np.array(predictions) + 1 

        tr = np.array(ground_truth)
        pre = np.array(predict1)
        error = (np.abs(tr - pre)/pre).mean()
        return error
    my_score = make_scorer(my_score_, greater_is_better=False) #自定义打分函数
    
    clf = grid_search.GridSearchCV(estimator=GradientBoostingRegressor(),
                                   param_grid={'loss': ['ls', 'huber'],
                                               'learning_rate': [0.1, 1],
                                               'n_estimators': [200,400,600,800],
                                               'max_depth': [3, 5, 7, 9]},
                                   cv = cv,
                                   scoring= cv_score) #scoring see: http://scikit-learn.org/stable/modules/model_evaluation.html
    clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values)
    classifier = clf.best_estimator_    
    best_scores = []
    output = []
    change_flag = para_search_step
    for i in range(len(test)):
        change_flag = change_flag-1
        if change_flag == 0:
            change_flag = para_search_step
            print 're-search best para, please wait ...'
            clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values) #x,y
            classifier = clf.best_estimator_
        best_scores.append(clf.best_score_)    
        output.append(classifier.predict(test.iloc[[i]][test.iloc[[i]].columns[1:]].values))
        train = pd.concat([train,test.iloc[[i]]]) #更新train
        print 'Training Set sizes:', len(train), 'score: ', clf.best_score_
    pre = pd.DataFrame(output,columns=['predicted'])
    pre.index = test.index
    df_re = pd.concat([test[[test.columns[0]]],pre],axis =1)
    df_re = df_re.rename(columns = {test.columns[0]:'observed'})
    df_re.index = pd.to_datetime(df_re.index)

    scores_re = pd.DataFrame(best_scores,columns=[cv_score])
    scores_re.index = test.index
    return df_re, scores_re
	
	
def LinearSVR_RFE_train_predict_rolling(dff, roll_begin_index = 51, 
                                        para_search_step = 20, 
                                        cv_score = 'neg_mean_absolute_error', 
                                        cv = 5,
                                        random_state=300):
    #dff first column is y, other columns is x1,x2,x3...
    
    org_train = dff.iloc[:roll_begin_index,] #第一条数
    org_test = dff.iloc[roll_begin_index:,]

    
    #特征选择：
    estimator = LinearSVR(random_state = random_state)
    #estimator = SVR(kernel="linear")
    selector = RFECV(estimator= estimator, step=1, cv=cv,
              scoring=cv_score) 
    
    X = org_train[org_train.columns[1:]]
    y = org_train[org_train.columns[0]]
    selector = selector.fit(X.values, y.values)
    rfe_se = pd.DataFrame(selector.support_,index = X.columns)
    select_index = rfe_se[selector.support_].index
    train =  org_train[org_train.columns[0]].to_frame(name = 'Y').join(org_train[select_index])
    test =   org_test[org_test.columns[0]].to_frame(name = 'Y').join(org_test[select_index])


    best_scores = []
    output = []

    clf = grid_search.GridSearchCV(estimator= estimator,
                             param_grid={
                                         'C': [0.1, 2,4,8],
                                         'epsilon':[0.1],
                                         'loss':['epsilon_insensitive','squared_epsilon_insensitive']
                                        },
                             cv = cv,
                             scoring= cv_score) #scoring see: http://scikit-learn.org/stable/modules/model_evaluation.html

    clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values)
    classifier = clf.best_estimator_
        
    change_flag = para_search_step
    for i in range(len(test)): 
        change_flag = change_flag-1
        if change_flag == 0:
            change_flag = para_search_step
            print 'fliter features and re-search best paras, please wait ...'
            
            #特征选择：
            estimator = SVR(kernel="linear")
            selector = RFECV(estimator= estimator, step=1, cv=cv,
                      scoring=cv_score) 
            
            X = org_train[org_train.columns[1:]]
            y = org_train[org_train.columns[0]]
            selector = selector.fit(X.values, y.values) #特征选择
            rfe_se = pd.DataFrame(selector.support_,index = X.columns)
            select_index = rfe_se[selector.support_].index
            train =  org_train[org_train.columns[0]].to_frame(name = 'Y').join(org_train[select_index])
            test =   org_test[org_test.columns[0]].to_frame(name = 'Y').join(org_test[select_index])

            #参数寻优
            clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values) #x,y
            classifier = clf.best_estimator_
            


        best_scores.append(clf.best_score_)    
        output.append(classifier.predict(test.iloc[[i]][test.iloc[[i]].columns[1:]].values)) #预测
        
        org_train = pd.concat([org_train,org_test.iloc[[i]]]) #更新org_train
        
        print 'Training Set sizes:', len(train), 'score: ', clf.best_score_, clf.best_params_

    pre = pd.DataFrame(output,columns=['predicted'])
    pre.index = test.index
    df_re = pd.concat([test[[test.columns[0]]],pre],axis =1)
    df_re = df_re.rename(columns = {test.columns[0]:'observed'})
    df_re.index = pd.to_datetime(df_re.index)

    scores_re = pd.DataFrame(best_scores,columns=[cv_score])
    scores_re.index = test.index
    return df_re, scores_re
	
def RBF_SVR_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5, pcc = 0.1):
    #dff first column is y, other columns is x1,x2,x3...
    
    org_train = dff.iloc[:roll_begin_index,] #第一条数
    org_test = dff.iloc[roll_begin_index:,]

    
    #特征选择：
    pcc_tr = org_train.corr()
    select_index = pcc_tr[pcc_tr[pcc_tr.columns[0]] > pcc].index
    train = org_train[select_index]
    test = org_test[select_index]

    #自定义打分函数
    def my_score_(ground_truth, predictions):
        predict1 = np.array(predictions) + 1 

        tr = np.array(ground_truth)
        pre = np.array(predict1)
        error = (np.abs(tr - pre)/pre).mean()
        return error

    my_score = make_scorer(my_score_, greater_is_better=False) #自定义打分函数

    best_scores = []
    output = []
    regr_2 = SVR()

    clf = grid_search.GridSearchCV(estimator= regr_2,
                             param_grid={'kernel': ['rbf'], 
                                         'gamma': [0.0005, 0.001, 0.01, 0.1,0.5,1],
                                         'C': [0.1, 2,4,8,20,200],
                                         'epsilon':[0.0005, 0.001,0.01, 0.1]
                                        },
                             cv = cv,
                             scoring= cv_score) #scoring see: http://scikit-learn.org/stable/modules/model_evaluation.html

    clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values)
    classifier = clf.best_estimator_
    change_flag = para_search_step
    for i in range(len(test)): 
        change_flag = change_flag-1
        if change_flag == 0:
            change_flag = para_search_step
            print 'fliter features and re-search best paras, please wait ...'
            
            #特征选择：
            pcc_tr = org_train.corr()
            select_index = pcc_tr[pcc_tr[pcc_tr.columns[0]] > pcc].index
            train = org_train[select_index]
            test = org_test[select_index]
            
            clf.fit(train[train.columns[1:]].values, train[train.columns[0]].values) #x,y
            classifier = clf.best_estimator_
            


        best_scores.append(clf.best_score_)    
        output.append(classifier.predict(test.iloc[[i]][test.iloc[[i]].columns[1:]].values)) #预测
        
        org_train = pd.concat([org_train,org_test.iloc[[i]]]) #更新org_train
        
        print 'Training Set sizes:', len(train), 'score: ', clf.best_score_, clf.best_params_

    pre = pd.DataFrame(output,columns=['predicted'])
    pre.index = test.index
    df_re = pd.concat([test[[test.columns[0]]],pre],axis =1)
    df_re = df_re.rename(columns = {test.columns[0]:'observed'})
    df_re.index = pd.to_datetime(df_re.index)

    scores_re = pd.DataFrame(best_scores,columns=[cv_score])
    scores_re.index = test.index
    return df_re, scores_re