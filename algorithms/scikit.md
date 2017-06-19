# Machine Learning Algorithm Recipes in scikit-learn

>  本文是原文的简略版笔记，英文原版[在此](http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)

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

   安装命令行： `xcode-select --install`。安装完成后可用`xcode-select -p`检验是否成功。有些情况下需要这样来同意使用条款`sudo xcodebuild -license`。

2. 安装[Macports](https://www.macports.org/)：包管理工具。

   安装成功后先进行更新升级`sudo port selfupdate`。

3. Install SciPy Libraries：

   1. 安装python3.5： `sudo port install python35`

   2. 将其设置为默认的版本：`sudo port select --set python python35 sudo port select --set python3 python35`

   3. 安装SciPy及其依赖：`sudo port install py35-numpy py35-scipy py35-matplotlib py35-pandas py35-statsmodels py35-pip`

      PS: 为了保证pip是默认使用的： `sudo port select --set pip pip35·`

   4. 用pip安装scikit-learn: `sudo pip install -U scikit-learn`

   5. ​

4. Install Deep Learning Libraries

   1. 安装Theano: `sudo pip install theano`
   2. 安装TensorFlow: `sudo pip install tensorflow`
   3. 安装Keras: `sudo pip install keras`