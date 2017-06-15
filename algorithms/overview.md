## A Tour of Machine Learning Algorithms

>  本文是原文的简略版笔记，英文原版[在此](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

## 算法分类

可以从两种维度来考虑和分类可能遇到的算法

1. 根据学习方式来分类
2. 根据形式和功能上的相似性来分类

### 根据学习方式分类

根据算法与经验和环境等等输入数据的交互的不同，一个算法可以有很多种建模方式。通常情况下在一些机器学习和人工智能的教科书里会认为学习是说算法能自己适应。目前仅有几种主流的学习方式和学习模型。这种算法组织方式很有用，因为这让你充分思考你问题的输入数据的角色，模型准备过程，然后选择最合适的方式来拿到最好的结果。

#### 有监督学习 Supervised Learning

输入数据为训练数据，有一个已知的标签或结果。用来做预测的模型是在训练过程中产生的，这个模型需要在预测错误的时候进行自我修正。模型在具有一定的准确性以后停止训练过程。

典型的应用场景是分类和回归。典型的算法包括逻辑回归和反向传播神经网路算法。

#### 无监督学习 Unsupervised Learning

输入数据没有标签和已知结果。模型在推断输入数据的结构时产生。这种方法使用数学方法系统的处理减少冗余，也可呢能根据相似性来组织数据。

典型的应用场景有聚类，降维，相关规则学习。典型的算法包括先验算法和k均值算法。

#### 半监督学习 Semi-Supervised Learning

输入数据是有标签和无标签样本的混合。有一种预测问题，模型必须学习数据的结构来组织数据并做出预测。

典型的应用场景是分类和回归。典型的算法是其他能对无标签数据做出假设的弹性方法的延伸。

![算法总览](overview.png)

### 根据相似性分类

我们经常根据算法的功能(怎么计算)将算法进行分类。比如说，基于树的算法和基于神经网络的算法。但是仍然有一些算法可以分到几个类目下，比如学习矢量量化（LVQ，Vector Quantization ）算法，既是神经网络算法也是基于实例的算法。也有一些名字相同的类目用于不同的场景比如回归和聚类。

#### 回归算法

回归关注变量之间关系的建模。通过模型提供预测错误的度量进行不断迭代重定义变量之间的关系来实现。回归广泛的应用于统计学，已经被并入统计机器学习。数据样本一般是一些离散的值，回归算法试着推测出这一系列连续值属性。主流的回归算法有：

-   普通最小二(OLSR, Ordinary Least Squares Regression)
-  线性回归 Linear Regression
-  逻辑回归 Logistic Regression
-  多元分析回归(逐步回归) Stepwise Regression
-  多元自适应回归样条 (MARS, Multivariate Adaptive Regression Splines)
-   本地散点平滑估计(LOESS, Locally Estimated Scatterplot Smoothing)

#### 基于实例的算法

基于实例学习解决的是决策问题。需要提供对于训练集来说重要的或者对模型来说必须的实例或者样本。这种算法一般会建立一个样本数据的数据库，将新数据与数据库中的数据对比衡量相似性找到最佳匹配做出预测。算法关注存储实例的展示和相似性衡量。主要有：

-  k邻近算法(kNN, k-Nearest Neighbor )
-  学习矢量量化(LVQ,  Learning Vector Quantization)
-  自组织映射模型 (SOM，Self-Organizing Map)
-  局部加权学习 (LWL， Locally Weighted Learning)

#### 正则化算法

正则化算法用于对过度拟合或者复杂度太高的模型进行简化，提高模型通用性的方法的延伸。正则化算法主要有：

-  岭回归（Ridge Regression）
-   最小绝对收缩与选择算子 (LASSO, Least Absolute Shrinkage and Selection Operator)
-  弹性网络 Elastic Net
-  最小角回归 (LARS, Least-Angle Regression )

#### 决策树算法

贝叶斯算法是用贝叶斯定理进行分类和回归的方法，主要有：

-  朴素贝叶斯 Naive Bayes
-  高斯朴素贝叶斯 Gaussian Naive Bayes
-  多项式朴素贝叶斯 Multinomial Naive Bayes
-   平均1-相依估计量(AODE， Averaged One-Dependence Estimators)
-  贝叶斯信念网络 (BBN,  Bayesian Belief Network)
-  贝叶斯网络 Bayesian Network (BN)

#### 聚类算法

聚类算法和回归类似，解决的是问题或方法的归类。聚类算法由比如基于质心的或者是分层的建模方式组成。为了达到将所有数据最好的归纳到尽可能大的分组里，所有的方式都由数据的内在结构决定。主流算法有:

-  K平均值 k-Means
-  K中位数 k-Medians
-   期望最大化 (EM, Expectation Maximisation)
-  层次聚类 Hierarchical Clustering

#### 关联规则学习算法

关联规则学习算法提取最合理的数据集中变量之间关系的规则。这些规则能帮助我们发现大数据集中重要的和有经济价值的关系。最主流的有：

-  先验算法 Apriori algorithm
-  Eclat算法 Eclat algorithm

#### 人工智能神经网络算法

人工智能神经网络算法是模仿生物学神网络的结构和功能的模型。研究一系列用于回归和分类问题的模式匹配。一般有：

-  感知机 Perceptron
-  反向传播 Back-Propagation
-  霍普菲尔德神经网络 Hopfield Network
-  径向基网络 (RBFN, Radial Basis Function Network )

#### 深度学习算法

深度学习是人工智能神经网络算法的现代化更新，研究大数量的廉价计算。关注的是构建更大规模更复杂的神经网络，许多方法都与有很少带标签数据的半监督学习问题有关。比如：

-  深度玻尔兹曼机 (DBM,  Deep Boltzmann Machine)
-  深度信念网络 (DBN,  Deep Belief Networks)
-   卷积神经网络(CNN, Convolutional Neural Network)
-  栈式自编码器 Stacked Auto-Encoders

#### 降维算法

与聚类算法类似，降维算法探索数据的内在的结构，但是数据量相对较小，是在一种无监督的情况下进行的。可以用来模拟数据维度，简化可以用于监督学习的数据。许多这种算法都被用于分类和回归。

-  主成分分析 (PCA, Principal Component Analysis)
-  主成分回归 (PCR, Principal Component Regression)
-  偏最小二乘回归 (PLSR, Partial Least Squares Regression)
-  萨蒙映射 Sammon Mapping
-  多维尺度分析 (MDS, Multidimensional Scaling)
-  投影追踪 Projection Pursuit
-  线性判别分析 (LDA,  Linear Discriminant Analysis)
-  混合判别分析  (MDA, Mixture Discriminant Analysis)
-  二次判别分析 (QDA, Quadratic Discriminant Analysis)
-  弹性判别分析 (FDA, Flexible Discriminant Analysis)

#### 融合算法

融合算法是由多个独立训练的弱模型组成的模型，模型做出的预测是通过一些方法融合出来的。关注的主要是如何将弱模型的结果结合，以及如何结合。主要有：

-  Boosting算法
-  Bagging算法 (Bootstrapped Aggregation )
-  AdaBoost是算法
-   泛化堆叠算法 (Stacked Generalization)
-   Gradient Boosting算法 (GBM, Gradient Boosting Machines)
-  Gradient Boosted Regression Trees (GBRT)
-  随机森林 Random Forest

#### 其他算法

目前列举出的算法并不包括机器算法实施过程中的算法，比如特征选择算法，算法精确率评估，性能评估等。

也不包括特定子领域的算法，比如：

-  Computational intelligence (evolutionary algorithms, etc.)

-  Computer Vision (CV)

-  Natural Language Processing (NLP)

-  Recommender Systems

-  Reinforcement Learning

-  Graphical Models

   ​

