## Welcome to TSP Pages

## **基于tsp的疾病预测案例**
#### **——手足口病发病人数的预测报告与说明**


**作者：申万祥**

**时间：2017年6月-2017年7月**

# **代码示例**
**[01. package_usage example: SARIMA model](https://shenwanxiang.github.io/tsp/SARIMA_eample.html)**

**[02. package_usage example: combination model](combination_model.html)**


# **案例说明目录**
[一、背景及介绍](#一-背景及介绍)

[二、数据来源与分析](#二-数据来源与分析)

[2.1 手足口病数据](#21-手足口病数据)

[2.2 天气及天气预报数据](#22-天气及天气预报数据)

[2.3 舆情数据](#23-舆情数据)

[三、预测方案及工具](#一-预测方案及工具)

[3.1 模型的架构](#_Toc486586302)

[1） 单模型构架思路](#_Toc486586303)

[2） 组合模型构架思路](#_Toc486586304)

[3.2 所用工具](#_Toc486586305)

[四、特征抽取与分析](#_Toc486586306)

[4.1 特征抽取、处理与分析](#_Toc486586307)

[4.2 特征选择](#_Toc486586308)

[五、模型评价指标](#_Toc486586309)

[5.1 模型预测的精确度评分](#_Toc486586310)

[5.2 模型的滞后性打分](#_Toc486586311)

[六、建立模型](#_Toc486586312)

[6.1 模型总体流程](#_Toc486586313)

[6.2 单模型预测](#_Toc486586314)

[6.3 组合网络模型预测](#_Toc486586315)

[1）：基于偏差的组合模型](#_Toc486586316)

[2）：输入输出组合模型](#_Toc486586317)

[七、结果与分析](#_Toc486586318)

[7.1 发病人数序列分析](#_Toc486586319)

[7.2 特征相关性分析](#_Toc486586322)

[7.3 模型建模过程](#_Toc486586323)

[7.4 模型结果与讨论](#_Toc486586324)

[1. 单模型结果](#_Toc486586325)

[2. 组合模型结果](#_Toc486586326)

[八、结论](#_Toc486586327)

[九、参考文献](#_Toc486586328)

[十、代码模块化与说明](#_Toc486586329)




# **一. 背景及介绍**

手足口病(Hand foot mouth disease, HFMD) 是由肠道病毒引起的常见传染病，在临床上以手、足、口腔疱疹为主要特征，故通称为手足口病，多发生于5岁以下的婴幼儿，所以常被称作"小儿手足口病"。HFMD全年均可有发病，但3~11月份多见，6~8月份为高峰期。传播速度极快，传播范围极广，具有周期流行的规律，一般2~3年流行一次。根据百度百科和维基百科，HFMD的发病潜伏期约1周（多为2～10天，平均3～5天）。

# **二. 数据来源与分析**

## **2.1 手足口病数据**
重庆市手足口病数据来自重庆市CDC部门，涉及到发病人数和就诊人数。其中就诊人数为医疗机构上报的就诊人数。涉及的发病地区主要有重庆，也有其他地区如北京，陕西，江西等，但是家庭住址全部为重庆地区的各个区县。原始的数据为2012年到2016年，分别为5个excel保存。因为原始的只有地区编码和住址编码，所以通过编码与地址对码，以及数据整合，最终生成一个csv文件，包含所有年份的数据，以及补充的报告区县，发病区县，家庭住区县（文件名为：**szk\_data.csv**）
## **2.2 天气及天气预报数据**
重庆市的天气及天气预报数据通过爬虫获取，文件名为（**重庆实际天气.csv，重庆天气预报.csv**）。天气数据分为白天和夜间的数据，在后面的特征转换盒抽取中，将天气描述性文字量化，将白天和夜间平均。
## 2.3 **舆情数据**
目前的舆情数据主要是百度指数（文件名：**baidu\_zhishumobile.txt，baidu\_zhishupc.txt**），由于该指数缺失值、奇异值过多，所以本次建模预测的时候暂时没有采用这部分数据，但是数据经过整合、填充，目前最新的数据文件名为：。

# **三. 预测方案及工具**

## **3.1 模型的架构**
所有的模型采用滚动预测的方法，进行一步预测（前一周预测后一周的）。所以随着滚动，训练集数据会逐渐增加。主要采用单模型和组合模型进行滚动预测。
### **1） 单模型构架思路**
单模型包括自动滞后一周模型（EWA模型），这是参照基准模型。最终的模型的效果应该远远比基准的效果要好，否则没有多大意义。其次是季节性ARIMA模型和非季节性ARIMA模型，这个模型利用时序数据，加入或者不加天气因子，以及天气预报因子。再次是有监督的SVR模型，GBDT模型以及DT-Adaboost模型，该模型加入的特征包括：发病人数本身的滞后x周，n阶差分滞后x周，气象特征滞后x周，时间特征不做任何滞后，可以看作是时效特征。所有单模型构建过程如图1。

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.001.png)

图1. 单模型构建思路流程
### **2） 组合模型构架思路**
组合模型思路大致分为两类，第一类是最终的输出=模型1的结果+模型2的结果，第二类是最终的输出=模型2的输出（其中模型1的输出是模型2的输入）。因此，第一类组合模型可以称为**基于偏差的组合模型**，这种方法主要是：模型2预测模型1的偏差，最后的结果就是模型1的结果加上模型2的结果。第二类组合模型可以称为**输出输入组合模型**，这种方法主要是：模型2的特征来源包括了模型1的输出。两类组合模型的特征输入如下图2所示

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.002.png)

图2. 组合模型构建思路流程
## **3.2 所用工具**
所有数据分析，处理，特征选择，构建模型，可视化和打分函数都用python语言，经过后续封装，可以直接调用**swx\_ts\_prediction**这个包使用，运行环境为python 2.7，其中包括**feature processing**, **evaluation**, **model**, **visualization**这四个小模块。该包依赖的主要的库有：pandas，numpy，scipy，statsmodels，sklearn，matplotlib和seaborn。调用该包时，确保加入环境变量，可以通过sys.path.append(r"D:\...")实现，详细信息请阅读readme.rst文件。

# **四、特征抽取与分析**

## **4.1 特征抽取、处理与分析**
* 1）真实天气及天气预报数据包括Categorical和Numerical数据，其中Categorical数据分解为雨天，晴天，云天，雷天，雾天，阴天，雪天，然后根据各种天气有描述性程度，附上不同的权重。Numerical数据主要是整合，差分和滞后；

* 2）手足口病发病人数数据，主要采用差分，滞后进行特征抽取。差分滞后由函数，其中对应的时间分解为年、月、日、周、季这5个特征；

* 3）舆情数据主要是Numerical数据，但是有缺失值，主要是采用前后一天数值进行缺失补充。

抽取到以上特征之后，进行差分滞后，单纯滞后，对特征做分布分析和相关性分析，采用skew方法将skewness大于0.75的取以对数，使得数据更加近似成为正态分布。对于高度线性相关(相关系数大于0.95)的特征保留其中的一个
## **4.2 特征选择**
1) 去除斜方差小于0.8的特征；

2) 单变量选择法：利用皮尔逊相关系数筛选线性相关性较大的特征（与训练集合数据的发病人数相关性大于0.2），此外，一些完全线性不相关的但可能是导致疾病的因素的特征也被添加进来；

3）在线性SVR模型中采用循环递归消除法选择特征（RFE）：具体做法是：反复的构建模型（SVR回归模型）然后交叉验证选出最好的特征（根据系数来选），把选出来的特征放到一遍，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。

# **五、模型评价指标**

## **1. 模型预测的精确度评分**

* 1）. 平均绝对百分偏差率（MAPE）：MAPE=(∑((|X-Y|)/X) \*100%)/N

* 2）. 平均绝对偏差（MAE）：MAE= ∑|X-Y| /N

* 3） 赤池信息量准则（AIC）和贝叶斯信息准则（BIC）：AIC = 2\*log-Likelihood+2\*K； BIC = 2\*log-Likelihood + K\*log(N)（K为参数个数）


## **2. 模型的滞后性打分**

此外，为了衡量模型在未来预测的上升或者下降走势的准确率，以及在疾病人数上涨的场景的预警率，根据上升和下降，将变化转换为分类问题的评价指标，包括以下4个指标（越高越好）：

* 1） 波峰，波谷皮尔逊相关系数（PCCP，PCCV）：波峰活波谷处皮尔逊相关系数的平均值；

* 2） 一步预测跌涨准确率（ACC）：ACC = (TN + TP）/ (TN+TP+FN+FP)

* 3） 上升趋势提起预警率（SEN）：SEN = TP / (TP + FN)

* 4） 下降趋势提前预警率（SPE）：SPE = TN / (TN + FP)


# **六、建立模型**

## **6.1 模型总体流程**
* 1) 使用滑动窗口+回归模型的方式完成建模，对过去K周数据训练建立回归模型，预测下一周趋势；
* 2) 训练窗口随时间持续滑动(训练集逐渐增加)，以保证模型的时效性；
* 3) 每次增加一定步长的训练集样本，就对模型的最佳参数进行格点搜索，更新模型的最佳参数（对于ARIMA模型，采用BIC，AIC作为参数寻优的指标，对于其他监督性学习模型，5-折交叉验证训练集，然后采用MAE或者R2作为寻优标准）。
## **6.2 单模型预测**
* 1）：EWA平移一周：将发病人数平移一周；
* 2）：SARIMA单模型：不加任何外部因子，直接采用病例数作为输入、输出，优化的参数有p, d, q；
* 3）：SARIMA单模型加外部因子：分别添加滞后的真实和预报的天气因子，优化的参数有p, d, q,和P, D, Q,s；
* 4）：SVR单模型加外部因子：除了滞后、差分的天气因子，还添加时间因子，病例数滞后因子，病例数差分滞后因子，SVR中分别采用RBF非线性核函数和线性核函数，优化的参数有：C，epsilon，gamma（RBF核），loss（线性核）；
* 5）：GBDT单模型加外部因子：GBDT外部因子和SVR相同,GBDT模型每一步主要调节loss，learning\_rate，n\_estimators，max\_depth这四个参数；
* 6）：Adaboost模型加外部因子：外部因子和SVR相同，模型使用DecisionTreeRegressor作为单个叠加器，优化的参数有：learning\_rate，n\_estimators，max\_depth，loss。
## **6.3 组合网络模型预测**
### **1）：基于偏差的组合模型**
EWA偏差组合模型：预测EWA与真实值的差异，然后加上EWA的预测值，为最终的输出，组合模型的类型有：EWA+SVR，EWA+Adaboost，EWA+GBDT；

SARIMA偏差组合模型：预测SARIMA与真实值的差异，然后加上SARIMA的预测值，根据EWA偏差组合模型结果，只做了SARIMA+SVR的组合模型；

在以上偏差预测过程中，偏差项需要与因子做进一步的相关性分析，选取相关性较大的因子作为偏差项的输入。
### **2）：输入输出组合模型**
输出输入组合模型是：第一步：整合天气因子等作为一个相关性较强的特征，第二步，该特征结合单模型输出的结果，以及天气因子等一起作为输入得到一个最终的结果。初步采用的组合模型的方式为：

SARIMA + Adaboost + 天气因子——>SVR

主要是考虑到如果使用相同的组合模型，会造成误差的叠加，而使用不同的组合模型则会取长补短，最终用SVR作为最后模型是因为，SVR与其他方法相比，采用RFE特征选择算法，可以选则性的加入特征，从而提升预测精度。

# **七、结果与分析**

## **7.1 发病人数序列分析**
发病人数的数据分布如图6所示，数据是重庆发病人数整体的统计结果。发病人数呈现周期性变化，趋势分解可以看出周期约为180天，约为25~26周。
### ![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.003.png) **![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.004.png)**
### **图6. 序列分布                    图7.序列趋势分解**
## **7.2 特征相关性分析**
* 1）真实的天气特征滞后相关性见图5，真实天气特征滞后一周雨发病人数有明显的相关性上升趋势，说明前一周的天气对发病人数影响较大，其中，前一周的云量、雨天与发病人数呈显著正相关，而前一周的晴天和云天与发病人数呈显著负相关，对应前一周的平均湿度呈显著负相关（p<<0.001）。与真实天气相比，预报因子相关性较弱，且滞后的周数较长，说明天气预报相对于真实天气有滞后性，可能的原因是天气预报相对于真实的天气也有滞后性。

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.005.png)![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.006.png)

![D:\Users\SHENWANXIANG533\Desktop\下载 (9).png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.007.png)![D:\Users\SHENWANXIANG533\Desktop\下载 (8).png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.008.png)

图 5 真实天气因子（真实和预报）特征与发病人数的滞后相关性

以上结果说明天气特征对手足口病爆发的导火线性因素，而手足口病爆发呈现周期性爆发的结果，根本的原因可能是因为气象因子的周期性结果导致的。同时说明，天气因素导致的手足口病人数增加，潜伏期为1周，而且这与手足口病从患病开始的潜伏期大约一致。

* 2）发病人数序列自相关性和偏差相关性见图6和图7：自相关和偏差相关性都是周期性衰减，自相关的最强负相关约为12~13周前数据，最强正相关约为24~26周前的数据，差分后最强的相关性为：差分11周，滞后一周的数据，相关系数都在0.8以上（p << 0.001）。

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.009.png)![D:\Users\SHENWANXIANG533\Desktop\下载 (3).png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.010.png)

图6. 序列不同滞后自相关           图7.序列不同差分后自相关

* 3）舆情因子的相关性分析: to be done

## **7.3 模型建模过程**
* 1）SARIMA单模型：SARIMA模型的建立，在模型选择的标准方面，采用训练集中最小的BIC和AIC值来选择模型，即确定p,d,q和P,D,Q,其中因为手足口病小周期接近12~13周，所以先尝试使用s为13，发现取12效果较好，所以后续优化就固定了s为12，然后优化p,d,q。因为加入不同的训练集，最优的模型可能不一样，所以设置了步长为30（因为SARIMA速度较慢，所以步长没有选择为1），选择一次模型。图8显示了建模过程中，选择不同模型的时候AIC和BIC的变化，可以看得出，随着训练集的增加，最小的BIC和AIC也逐渐变小。

![D:\Users\SHENWANXIANG533\Desktop\下载.png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.011.png)

图8. 滚动预测最小的AIC和BIC的变化

* 2）其他监督学习单模型: 其他监督性学习模型在参数优化的过程中，打分函数可以选择R平方或者负的MAE，原始代码默认为负的MAE。在随着滚动窗口的前进，训练集的增加，5折交叉验证的R2都会越来越高，如图9所示（预测偏差组合模型），MAE越来越低，其变化如图10所示（输入输出组合模型）。

![D:\Users\SHENWANXIANG533\Desktop\下载 (1).png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.012.png)

图8. 偏差组合模型中单模型的滚动预测R2的变化

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.013.png)

图10. 输入输出组合模型滚动预测MAE的变化


## **7.4 模型结果与讨论**
### **1. 单模型结果**
* 1）模型的最终对比结果使用的是2014到2016年的预测结果，见表1。其中，效果最好的单模型是SARIMA+部分天气因子单模型，MAPE为0.144，ACC可以达到0.73，PCCP可以到0.79。相比较加入全部天气因子，加入部分因子使得结果改善（0.144 vs. 0.156），去掉的因子就是协方差小于0.8的特征，这说明在SARIMA中，使用协方差过滤因子是非常有效的。

* 2）其他单模型中，RBF-SVR效果最好，在RBF-SVR中采用了特征选择，每次滑动的时候选择PCC大于0.0001的特征。选取相关性较小的特征是因为同时考虑非线性相关特征。在其他单模型中，过滤协方差较小的特征提升不明显甚至变差。这说明在其他监督性学习模型中（ADA, GBDT, RFE-LSVR），没有必要首先过滤特征，因为这些方法自带有特征选择；在SVR中，使用RBF核函数性能比线性核函数性能好很多（0.177 vs. 0.183）。值得注意的是，这些单模型的MAPE效果虽然没SARIMA的好，但是在滞后性指标上（表1），这些模型的效果整体较好，说明这些单模型有助于改善模型的滞后性（可参见图11，图12）。

以上结果和文献报道较为一致[1]，RBF-SVR的性能也比较好。同时文献报道了SARIMA结合卡尔曼滤波（Kalman filter）效果最好。其他文献[2]中也有相似的结果，说明卡尔曼滤波的确能改善SARIMA模型，后续可以探索和尝试这种方法。

![D:\Users\SHENWANXIANG533\Desktop\下载 (9).png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.014.png)

**图11. SARIMA+weather单模型的预测效果**

![D:\Users\SHENWANXIANG533\Desktop\下载 (10).png](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.015.png)

**图12. Adaboost单模型的预测**

### **2. 组合模型结果**
在基于偏差的组合模型中，**初始模型（EWA或者SARIMA）总是加上SVR的效果好**, 并且SVR相比于其他模型，速度最快。目前该类组合模型最好的是SARIMA+SVR组合，和单模型SARIMA+部分天气因子一样都是0.144。** SARIMA+SVR组合是基于SARIMA不加任何天气因子做得结果（0.156），后续可以基于SARIMA+部分天气因子，然后加上SVR提高预测精度，其中SVR可以选择舆情因子等；

在基于输出输入组合模型中，SARIMA + ADA +SVR是所有模型效果中最好的模型。MAPE可以达到0.132（三年的预测效果见图13），同时滞后性也得到了很大的改善。分析原因是：使用了SARIMA的输出，确保了结果不会太差，使用Adaboost整合的天气因子，改善滞后性，同时加入天气因子，再利用RFE特征选择算法每次有选择利用不同的特征。这种三组合模型改善的主要原因是集成了以上单模型的各自优点。

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.016.png)

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.017.png)

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.018.png)

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.019.png)

图13 SARIMA + ADA +LSVR整体金额分年的预测效果


|<p></p><p></p><p></p><p></p><p></p><p>单</p><p>模</p><p>型</p>|Model|MAPE|ACC|SEN|PCCP|PCCV|TIME(s)|
| - | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
||EWA|0.188|0.684|0.680|0.527|0.954|--|
||SARIMA|0.158|0.68|0.69|0.56|0.96|130|
||SARIMA+ forecast|0.165|0.645|0.667|0.54|0.954|145|
||**SARIMA+ weather**|**0.144**|**0.730**|**0.79**|**0.79**|**0.954**|**145**|
||SVR(Linear,RFE)|0.183|0.697|0.68|0.48|0.96|30|
||SVR(RBF)|0.177|0.71|0.70|0.71|0.96|40|
||GBDT|0.211|0.736|0.76|0.63|0.945|120|
||ADA|0.202|0.72|0.786|0.53|0.93|180|
|<p>组</p><p>合</p><p>模</p><p>型</p>|EWA + GBDT|0.159|0.64|0.64|0.48|0.92||
||EWA + ADA|0.156|0.65|0.65|0.53|0.96||
||EWA + SVR|0.145|0.66|0.65|0.55|0.97||
||SARIMA + SVR|0.144|0.77|0.77|0.64|0.96||
||**SARIMA + ADA +LSVR**|**0.132**|**0.71**|**0.75**|**0.79**|**0.954**||


**表 1 单模型和组合模型预测结果对比（预测的时间段为：2014~2016年）**

【注】：组合模型中，浅色的是基于偏差的组合，深色的是基于输出、输入的组合



# **八、结论**
通过使用单模型，组合模型的对比，发现：

* 1. 在单模型中，他们各有利弊。SARIMA的预测结果的偏差总是较小(最好的是0.144)，但是滞后性比较严重，其他模型的预测偏差较大，但是滞后性相比于SARIMA有所改善；
* 2. 采用组合模型集成他们各自优缺点，发现SARIMA+ADA+LSVR效果最好，MAPE为0.132，此外，SARIMA+SVR效果也不错，MAPE为0.144。以上结果说明组合模型有利于模型精度的提升，同时为了避免模型之间误差的叠加，应该采用不同的模型取长补短来组合。


# **九、参考文献**
* 1  Lippi M, Bertini M, Frasconi P. Short-term traffic flow forecasting: An experimental comparison of time-series analysis and supervised learning[J]. IEEE Transactions on Intelligent Transportation Systems, 2013, 14(2): 871-882.
* 2  Liu H, Tian H, Li Y. Comparison of two new ARIMA-ANN and ARIMA-Kalman hybrid methods for wind speed prediction[J]. Applied Energy, 2012, 98: 415-424.


# **十、代码模块化与说明**
代码主要分为4个部分：

![](Aspose.Words.f5ca0ec1-d824-487b-bde8-28f707c56f29.020.png)

1. feature\_preprocessing模块：
   1) remove\_low\_variance\_features(data\_frame,var = 0.8)函数：去除方差较小特征，VarianceThreshold为0.8
   2) check\_skew\_log(df,alpha = 0.75)函数：将所有特征近似为正态分布，skewed阈值为0.75
   3) diff\_shift\_lag(df,diff\_lag = 10,shift\_lag = 2)函数：将时序滞后，差分，得到新的特征，diff\_lag为差分最大阶数，shift\_lag为滞后最大阶数
   4) series\_to\_supervised(data, n\_in=1, n\_out=1, dropnan=True)函数：将时序滞后n\_in次得到新的特征
2. model模块：
   1) grid\_search\_ARMA\_para(ts, pdqrange = (0,2))函数：用于SARIMA模型不加外部因子情形下的格点搜索，同时也可以搜索s，按需修改
   2) ARMA\_train\_predict\_rolling(ts\_log, roll\_begin\_index = 5, para\_search\_step = 20)函数：SARIMA不加外部因子模型，roll\_begin\_index表示起始滑动的索引，para\_search\_step为格点搜索的步长；
   3) grid\_search\_ARMA\_Ex\_para(ts,ex\_train, pdqrange = (0,2))函数，与grid\_search\_ARMA\_para相同，只不过是加了外部因子（有时间需要加工整合一下）
   4) ARMA\_Ex\_train\_predict\_rolling(ts\_log, ex\_vectors, roll\_begin\_index = 51, para\_search\_step = 20, pdqrange = (0,2))函数：加外部因子的SARIMA模型，ex\_vectors为外部因子
   5) my\_score\_(ground\_truth, predictions)函数：自定义打分函数，可以定义好加入模型中，在参数寻优时采用此打分函数，也可以采用默认的如r2，MAE等；
   6) Adaboost\_train\_predict\_rolling(dff, roll\_begin\_index = 51, para\_search\_step = 20, cv\_score = 'neg\_mean\_absolute\_error', cv = 5)函数：ADA模型的函数，cv\_score即为格点搜索打分函数，优化的参数请在源码中按需修改，这部分由于时间原因没有加进来。CV是交叉验证次数；
   7) GBDT\_train\_predict\_rolling(dff, roll\_begin\_index = 51, para\_search\_step = 20, cv\_score = 'neg\_mean\_absolute\_error', cv = 5)函数，与上面的函数相同，用于GBDT模型；
   8) LinearSVR\_RFE\_train\_predict\_rolling(dff, roll\_begin\_index = 51, para\_search\_step = 20, cv\_score = 'neg\_mean\_absolute\_error', cv = 5,random\_state=300)函数，用于线性SVR滚动预测，其中采用了RFE特征选择方法，如果换成RFE非线性核，暂时不支持特征选择；
   9) RBF\_SVR\_train\_predict\_rolling(dff, roll\_begin\_index = 51, para\_search\_step = 20, cv\_score = 'neg\_mean\_absolute\_error', cv = 5, pcc = 0.1)函数，SVR中采用RBF核，特征选择采用皮尔逊相关系数，pcc为0.1，默认每次滑动选择pcc大于0.1的特征；
3. evaluation模块：
   1) evaluate\_func(df, self = True, level = 0, threshold = 0)函数：返回MAPE,ACC,SEN等，self默认为True，即预测值和实际值都是自己和自己比；
   2) test\_stationarity(timeseries, window=365)函数：用于时间需的平稳性检测
   3) trend\_split(ts, freq = 52)函数：用于时间序列的趋势分解
   4) cal\_wave\_pcc(df, threshold\_up = '75%',threshold\_down = '25%')函数：用于计算波峰波谷相关性函数，threshold\_up为大于多少人为是波峰，threshold\_down为小于多少为波谷，默认为上中和下中位数；
4. visualization模块：
   1) plot\_feature\_import(dfu)函数：特征相关性排序图，dataframe的第一列为y
   2) plot\_diff\_shift(df,diff\_lag = 10,shift\_lag = 2,title = u'发病人数在滞后与差分后相关性')函数：时间序列滞后、差分后相关性图
   3) plot\_diff\_zhihou\_corr(dfx, dfy,diff = 2,lagmax = 10)：时间序列在与特征滞后、差分后的相关性；
   4) plot\_correlation\_map(df)：相关性矩阵热图
   5) plot\_zhihou\_corr(dfx, dfy, title,lag\_max = 30,kind ='line')：时间序列在与特征滞后的相关性；
   6) get\_p\_value(df1,df2,name)：获取相关性p值
   7) swx\_scatter\_matrix(frame, alpha=0.5, beta = 0.8)函数：相关性及特征分布矩阵图，alpha为透明度，beta为控制text的背景颜色

