# Caffe安装记

>  mac OSX 10.12，CPU-ONLY version

本来在tensorflow搞了一套了，但是要嵌入前端的话tensorflow有问题，还是要搞caffe版的，所以就入了caffe的坑了。。装这个东西装了一整天，好气哦，一定要记一下。

### caffe的依赖库

-  Homebrew: `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

- cmake: 官网[下载](https://cmake.org/download/)后直接安装即可

- python: 如果你电脑里有N个版本的python，一定要把不用的版本清理干净！！然后用Homebrew安装一个**2.7**版本的python，并且安装好对应的*numpy*。否则之后在安装完后会有`segment fault:11`错误，这就倒霉了。别问我为啥知道的，都是泪。

- 其他一些依赖，下面这种装法是最好的，千万不要作死用pip：

   ```shell
   for x in snappy leveldb gflags glog szip hdf5 lmdb homebrew/science/opencv;
   do
       brew uninstall $x;
       brew install --fresh -vd $x;
   done
   brew uninstall --force protobuf; brew install --with-python --fresh -vd protobuf
   brew uninstall boost boost-python; brew install --fresh -vd boost boost-python
   ```


-  CUDA & cuDNN

   首先，看一下电脑有没有N卡，木有就别折腾了。。总之我没有，所以装了也没用。直接装了caffe的CPU ONLY版本。

- 下载caffe源码，记得找1.0正式版！

   ```shell
   git clone https://github.com/BVLC/caffe.git
   cd caffe
   cp Makefile.config.example Makefile.config
   ```

### 进入正题

-  打开*Makefile.config*，修改以下内容：

   ```
   CPU_ONLY := 1
   # NOTE: this is required only if you will compile the python interface.
   # We need to be able to find Python.h and numpy/arrayobject.h.
   PYTHON_INCLUDE := /usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
   		/usr/local/Cellar/numpy/1.13.3/lib/python2.7/site-packages/numpy/core/include/numpy
   		
   # We need to be able to find libpythonX.X.so or .dylib.		
   PYTHON_LIB := /usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/lib

   # Uncomment to support layers written in Python (will link against Python libs)
   WITH_PYTHON_LAYER := 1
   ```

- 接下来可以开搞了

   ```
   mkdir build && cd build
   cmake ..
   ```

- 刹车！这里有个小坑。如果是装CPU ONLY版本，一定要去找到*build*下面的*CMakeCache.txt*和`CaffeConfig.cmake`，`cmake_install.cmake`搜索里面的`CPU`关键字，把对应的`OFF`都改成`ON`，这边config居然没用，也是神奇。

- 接下来就`make all`啦。如果报有`<sys/sysInfo.h>`缺失，别整了，换个caffe版本。这样应该就没有问题啦。

- 测试一下是否编译成功：`make runtest`。

- 都成功的话，将下面路径添加到`~/.bash_profile`，然后source一下。

   ```
   export PYTHONPATH=/path/to/caffe/dir/python:$PYTHONPATH
   ```

- 还是在`caffe/build`目录下，执行：`make pycaffe`，然后测试一下`make pytest`。

- 如果控制台报错:

   ```
   ImportError: dynamic module does not define module export function (PyInit__caffe)
   ```

   说明你的library路径不对，很可能是用了**3.X**的版本。检查一下*CMakeCache.txt*，然后`make clean`一下，再试一次。