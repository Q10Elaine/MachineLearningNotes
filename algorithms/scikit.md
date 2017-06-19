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

      -  为了保证pip是默认使用的： `sudo port select --set pip pip35`

   4. 用pip安装scikit-learn: `sudo pip install -U scikit-learn`

4. 安装深度学习库

   1. 安装Theano: `sudo pip install theano`
   2. 安装TensorFlow: `sudo pip install tensorflow`
   3. 安装Keras: `sudo pip install keras`

总结：历时2个小时安装成功所有工具。基本步步都是坑，耐心看一下报错，多用google，还是都能解决的。

### 试水- [鹫尾花数据集](https://archive.ics.uci.edu/ml/datasets/Iris)

Iris数据集是常用的分类实验数据集，是一类多重变量分析的数据集。数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

本程序直接使用pandas读取了UCI Machine Learning repository的[数据](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)。

PS: 如果你的电脑里有多个版本的python，而且用的是PyCharm，在Run的时候可能会遇到not a module name的问题。这时候在preference里面找project interpretor，设置成3.5版本的python即可。