# Machine Learning Algorithm Recipes in scikit-learn

scikit-learn提供Python语言实现的监督和无监督学习的算法。这个库基于*SciP*,包括：

-  **NumPy**:n维度数组基础包
-  **SciPy**: 科学计算的底层库
-  **Matplotlib**: 综合的2D/3D绘图
-  **IPython**: 交互控制
-  **Sympy**: 符号数学
-  **Pandas**: 数据结构和分析

### 准备工作

1. 安装XCode和命令行工具：

   目前系统版本是maxOS Sierra 10.12.5,要求XCode版本^8.2.1。在APP store下载XCode。

   安装命令行： `xcode-select --install`。安装完成后可用`xcode-select -p`检验是否成功。如果你装成功了，you are a lucky dog。我装的时候点完同意使用条款就没有然后了，Stack Overflow上的兄弟是这样解的：`sudo xcodebuild -license`。用空格翻到最下方，打agree同意后再打开XCode就可以了。

2. 安装[Macports](https://www.macports.org/)：包管理工具。

   安装成功后先进行更新升级`sudo port selfupdate`。

3. 安装 SciPy库：

   1. 安装python3.5： `sudo port install python35`

   2. 将其设置为默认的版本：`sudo port select --set python python35 sudo port select --set python3 python35`

   3. 安装SciPy及其依赖：`sudo port install py35-numpy py35-scipy py35-matplotlib py35-pandas py35-statsmodels py35-pip`

      -  如果你成功安装了，恭喜你。反正我没安装成功一个叫GMP的依赖。所以这个依赖是我自己手动安装的。

         从[官网](https://gmplib.org/)下载最新的bz2压缩包，在下载目录中`tar -jvxf gmp-5.1.0.tar.bz2`。进入目标文件夹，`./configure --enable-cxx`。别急，如果这步有问题，提示*could not find a working compiler, see config.log for details*。这时候说明XCode的命令行没找到，再这样`sudo xcode-select --switch /Library/Developer/CommandLineTools/`。这时候就能搞定了。然后`make`， 'make check'， 最后`sudo make install`就安装完成了。然后回到上面的命令继续安装。

      - 为了保证pip是默认使用的： `sudo port select --set pip pip35`

   4. 用pip安装scikit-learn: `sudo pip install -U scikit-learn`

4. 安装深度学习库

   1. 安装Theano: `sudo pip install theano`
   2. 安装TensorFlow: `sudo pip install tensorflow`
   3. 安装Keras: `sudo pip install keras`

总结：历时2个小时安装成功所有工具。基本步步都是坑，耐心看一下报错，多用google，还是都能解决的。

### 试水- [鸢尾花数据集](https://archive.ics.uci.edu/ml/datasets/Iris)

本程序直接使用pandas读取了UCI Machine Learning repository的[数据](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)。基本操作步骤共有四步。

1. 加载样本数据：scikit-learn自带一些标准的数据集，比如鸢尾花。Iris数据集是常用的分类实验数据集，是一类多重变量分析的数据集。数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。数据集一般是一个*n样本，m特性*的数组。

```python
from sklearn import datasets
iris = datasets.load_iris()
print(iris.data)
```

​	也可以使用pandas读取数据，并且同时可以得到很多统计数据

```python
import pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
```

2. 观察数据

   -  查看汇总数据：如果是用pandas导入数据，可以很方便的查看一些数据的情况。

      ``` python
      print(dataset.shape) 					# 获取样本条数和属性总数
      print(dataset.describe())    			# 打印出总数，均值，最大最小值之类的各种数据
      print(dataset.groupby('class').size())  # 打印出根据class关键字分类的数据集大小
      ```

   -  数据可视化

      ```python
      import matplotlib.pyplot as plt
      # 箱线图
      dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
      plt.show()
      # 各个维度的柱状图
      dataset.hist()
      plt.show() 
      ```

3. 选择算法：创建模型，评估精确度。

   1. 创建校验数据集：用统计方法估算模型的精确度。我们也需要将模型运用到未知数据上对其精确度做具体的估算。所以，我们要保留一些数据用于二次校验。一般是将数据集分成两部分，80%用于训练模型，20%作为校验数据集。

      ``` python
      # Split-out validation dataset
      array = dataset.values
      X = array[:,0:4]
      Y = array[:,4]
      validation_size = 0.20
      seed = 7
      X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
      ```

   2. 测试模型：我们并不知道哪种算法，配置对于当前的问题是最好的。由于从数据可视化的图中可以看出有些类在有些维度是部分线性分割的。所以就尝试以下6种算法。

   -  Logistic Regression (LR)
   -  Linear Discriminant Analysis (LDA)
   -  K-Nearest Neighbors (KNN).
   -  Classification and Regression Trees (CART).
   -  Gaussian Naive Bayes (NB).
   -  Support Vector Machines (SVM).

   这些算法中包含简单线性的(LR, LDA), 非线性的(KNN, CART, NB, SVM)。每次运行前需要重置随机数种子，保证每个算法的试用都是用的相同的数据分割，结果是可比较的。

   ``` python
   # Spot Check Algorithms
   models = []
   models.append(('LR', LogisticRegression()))
   models.append(('LDA', LinearDiscriminantAnalysis()))
   models.append(('KNN', KNeighborsClassifier()))
   models.append(('CART', DecisionTreeClassifier()))
   models.append(('NB', GaussianNB()))
   models.append(('SVM', SVC()))
   # evaluate each model in turn
   results = []
   names = []
   for name, model in models:
   	kfold = model_selection.KFold(n_splits=10, random_state=seed)
   	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
   	results.append(cv_results)
   	names.append(name)
   	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   	print(msg)
   ```

   ​

   3. 选择最佳模型：我们现在有6个模型和精确性估计。我们需要比较这些模型，选择精确度最佳的模型。根据第2步的操作，我们可以得到以下数据:

   ````
   LR: 0.966667 (0.040825)
   LDA: 0.975000 (0.038188)
   KNN: 0.983333 (0.033333)
   CART: 0.975000 (0.038188)
   NB: 0.975000 (0.053359)
   SVM: 0.981667 (0.025000)
   ````

   从数据中可以看出KNN算法的效果最好。

   4. 校验：用校验数据集检验模型精确性。设置一个校验数据集是很有必要的，以免训练集过度适应，或者数据泄露。这些都会导致过分乐观的结果。

   我们用KNN模型预测校验数据集，将结果汇总出来制作一个混淆矩阵和分类报告。

   ``` python
   # Make predictions on validation dataset
   knn = KNeighborsClassifier()
   knn.fit(X_train, Y_train)
   predictions = knn.predict(X_validation)
   print(accuracy_score(Y_validation, predictions))
   print(confusion_matrix(Y_validation, predictions))
   print(classification_report(Y_validation, predictions))
   ```

   可以看到有90%的精确度。从混淆矩阵中可以看出有3个错误预测。分类报告显示有了每一类的精确度，召回率，f1分和支持度。

   ``` python
   0.9

   [[ 7  0  0]
    [ 0 11  1]
    [ 0  2  9]]

                precision    recall  f1-score   support

   Iris-setosa       1.00      1.00      1.00         7
   Iris-versicolor   0.85      0.92      0.88        12
   Iris-virginica    0.90      0.82      0.86        11

   avg / total       0.90      0.90      0.90        30
   ```

   ​

   PS: 如果你的电脑里有多个版本的python，而且用的是PyCharm，在Run的时候可能会遇到not a module name的问题。这时候在preference里面找project interpretor，设置成3.5版本的python即可。

