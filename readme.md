## This is a package for time-series prediction and results visualization
### Author : shenwanxiang
### Email : shenwx13@tsinghua.edu.cn
### version : 0.1
### last update date : 2017/6/27

### Usage:
```python 
import sys
#and then add path: 
sys.path.append(r"D:\...")
```

* 1.	feature_reprocessing模块：
  * a)	remove_low_variance_features(data_frame,var = 0.8)函数：去除方差较小特征，VarianceThreshold为0.8
  * b)	check_skew_log(df,alpha = 0.75)函数：将所有特征近似为正态分布，skewed阈值为0.75
  * c)	diff_shift_lag(df,diff_lag = 10,shift_lag = 2)函数：将时序滞后，差分，得到新的特征，diff_lag为差分最大阶数，shift_lag为滞后最大阶数
  * d)	series_to_supervised(data, n_in=1, n_out=1, dropnan=True)函数：将时序滞后n_in次得到新的特征

* model模块：
  * a)	grid_search_ARMA_para(ts, pdqrange = (0,2))函数：用于SARIMA模型不加外部因子情形下的格点搜索，同时也可以搜索s，按需修改
  * b)	ARMA_train_predict_rolling(ts_log, roll_begin_index = 5, para_search_step = 20)函数：SARIMA不加外部因子模型，roll_begin_index表示起始滑动的索引，para_search_step为格点搜索的步长；
  * c)	grid_search_ARMA_Ex_para(ts,ex_train, pdqrange = (0,2))函数，与grid_search_ARMA_para相同，只不过是加了外部因子（有时间需要加工整合一下）
  * d)	ARMA_Ex_train_predict_rolling(ts_log, ex_vectors, roll_begin_index = 51, para_search_step = 20, pdqrange = (0,2))函数：加外部因子的SARIMA模型，ex_vectors为外部因子
  * e)	my_score_(ground_truth, predictions)函数：自定义打分函数，可以定义好加入模型中，在参数寻优时采用此打分函数，也可以采用默认的如r2，MAE等；
  * f)	Adaboost_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5)函数：ADA模型的函数，cv_score即为格点搜索打分函数，优化的参数请在源码中按需修改，这部分由于时间原因没有加进来。CV是交叉验证次数；
  * g)	GBDT_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5)函数，与上面的函数相同，用于GBDT模型；
  * h)	LinearSVR_RFE_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5,random_state=300)函数，用于线性SVR滚动预测，其中采用了RFE特征选择方法，如果换成RFE非线性核，暂时不支持特征选择；
  * i)	RBF_SVR_train_predict_rolling(dff, roll_begin_index = 51, para_search_step = 20, cv_score = 'neg_mean_absolute_error', cv = 5, pcc = 0.1)函数，SVR中采用RBF核，特征选择采用皮尔逊相关系数，pcc为0.1，默认每次滑动选择pcc大于0.1的特征；

* evaluation模块：
  * a)	evaluate_func(df, self = True, level = 0, threshold = 0)函数：返回MAPE,ACC,SEN等，self默认为True，即预测值和实际值都是自己和自己比；
  * b)	test_stationarity(timeseries, window=365)函数：用于时间需的平稳性检测
  * c)	trend_split(ts, freq = 52)函数：用于时间序列的趋势分解
  * d)	cal_wave_pcc(df, threshold_up = '75%',threshold_down = '25%')函数：用于计算波峰波谷相关性函数，threshold_up为大于多少人为是波峰，threshold_down为小于多少为波谷，默认为上中和下中位数；

* visualization模块：
  * a)	plot_feature_import(dfu)函数：特征相关性排序图，dataframe的第一列为y
  * b)	plot_diff_shift(df,diff_lag = 10,shift_lag = 2,title = u'发病人数在滞后与差分后相关性')函数：时间序列滞后、差分后相关性图
  * c)	plot_diff_zhihou_corr(dfx, dfy,diff = 2,lagmax = 10)：时间序列在与特征滞后、差分后的相关性；
  * d)	plot_correlation_map(df)：相关性矩阵热图
  * e)	plot_zhihou_corr(dfx, dfy, title,lag_max = 30,kind ='line')：时间序列在与特征滞后的相关性；
  * f)	get_p_value(df1,df2,name)：获取相关性p值
  * g)	swx_scatter_matrix(frame, alpha=0.5, beta = 0.8)函数：相关性及特征分布矩阵图，alpha为透明度，beta为控制text的背景颜色
