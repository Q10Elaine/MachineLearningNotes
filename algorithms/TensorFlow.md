# TensorFlow API

>  最近在学习用CNN做图像分类，学习的时候发现并没有一个很好的中文API文档。所以就把自己用到的API都沉淀一下。

## tf.nn

### tf.nn.conv2d  tensorflow/python/ops/gen_nn_ops.py

```python
import tensorflow as tf
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    name=None
)
```

根据四维的输入`input`和`filter`张量计算一个二维的卷积。四维的`input`张量的维度分别是`[batch, in_height, in_width, in_channels]`。`filter`过滤器/核张量的维度是`[filter_height, filter_width, in_channels, out_channels]`。这个函数会进行如下操作：

1. 将过滤器拍平成二维矩阵，维度是`[filter_height * filter_width * in_channels, output_channels]`。
2. 用一个虚拟的`batch, out_height, out_width, filter_height * filter_width * in_channels]`张量从`input`抽取图片块
3. 对于每个图片块右乘过滤矩阵和图片块向量。

也就是如下形式：

```
output[b, i, j, k] =
    sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                    filter[di, dj, q, k]

```

该函数必须有 `strides[0] = strides[3] = 1`. 对于大多数场景会有相同的水平和垂直的步长, `strides = [1, stride, stride, 1]`。参数详细介绍如下：

-  **input**: 必须为 `half`或`float32`类型的四维张量。
-  **filter**: 与`input`类型一致的四维张量，维度是`[filter_height, filter_width, in_channels, out_channels]`。
-  **strides**: 长度为4的1维的整数数组。在`input`的每个维度的滑动窗口的步长。
-  **padding**: 值为 `"SAME"`或` "VALID"`的字符串。使用的边距算法。
-  **use_cudnn_on_gpu**: 可选的 `bool`值，默认为 `True`。
-  **data_format**: 可选的值为 `"NHWC", "NCHW"`的字符串，默认为`"NHWC"`。指定输入和输出的数据格式。值为 "NHWC"时，数据格式为[batch, height, width, channels]。值为"NCHW"时，数据格式为 [batch, channels, height, width].
-  **name**: 操作名称，可选。

该函数返回一个与`input`类型和维度都相同的张量。

###tf.nn.max_pool(value, ksize, strides, padding, name=None)

对输入实行max pooling。pooling后会让之前的特征维数减少，训练参数减少，泛化能力加强，进而防止过拟合。    其中特征维数的减少并不会让之前的特征丢失。

1. 为模型引入invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(伸缩)。
2. 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力。

##### Args:

-  **value**: 一个四维张量， `[batch, height, width, channels]` ，并且类型是 `float32`, `float64`, `qint8`, `quint8`, `qint32`.
-  **ksize**: 一个长度>= 4的整数数组. 是每个维度的输入张量的窗口大小。
-  **strides**: 一个长度>= 4的整数数组. T每个维度的滑动窗口的步长。
-  **padding**: 值为 `"SAME"`或` "VALID"`的字符串。使用的边距算法。
-  **name**: 操作名称，可选。

该函数返回一个与`value`类型和维度都相同的张量。

### tf.nn.lrn(input, depth_radius, bias, alpha, beta, name)

4D的输入`input`被看成一个3D的数组和1D的向量(最后一个维度)，每一个向量都会单独进行归一化。 对于每一个向量，每个部分会除以加权平方的`depth_radius`。

```
sqr_sum[a, b, c, d] =
    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias + alpha * sqr_sum) ** beta
```

#### Args:

-  **input**: 一个类型为 `float32`, `half`的四维张量。
-  **depth_radius**:一个可选的 `int`。默认是5。
-  **bias**: 可选的`float`. 默认是 `1`. 一个小偏移，避免除0
-  **alpha**: 可选的 `float`. 默认是 `1`. 一个伸缩因子，通常是正数。
-  **beta**: 可选的 `float`. 默认是 `0.5`. 一个指数.
-  **name**: 操作名称，可选。

#### Returns:

该函数返回一个与`input`类型和维度都相同的张量。

### tf.nn.batch_normalization 

```python
batch_normalization(
    x,
    mean,
    variance,
    offset,
    scale,
    variance_epsilon,
    name=None
)
```

详情可见 http://arxiv.org/abs/1502.03167. BN的优点是允许网络使用较大的学习速率进行训练，加快网络的训练速度（减少epoch次数），提升效果。用平均值和方差对张量进行归一化，可以用一个伸缩因子γ和一个偏移量β。具体参考：
$$
\frac{\gamma(x-\delta)}{\sigma}-\beta
$$
`mean`, `variance`, `offset` 和`scale` 都是以下两种结构中的一种。

-  大多数情况下，这些参数应该与输入 `x`有相同的维度。在这种情况下  `mean` 和 `variance` 是训练过程中`tf.nn.moments(..., keep_dims=True)` 的输出，或者是建模过程中的平均数
-  通常情况下，`depth`维度是输入`x`的最后一个维度，比如全连接层的 `[batch, depth]` 以及卷积层的 `[batch, height, width, depth]` 。`mean` 和 `variance` 是训练过程中的`tf.nn.moments(..., keep_dims=False)`的输出，或者是建模过程中的平均数

#### Args:

-  **x**:任意维度的输入张量。
-  **mean**: 平均张量
-  **variance**: 方差张量
-  **offset**: 偏移量β。
-  **scale**: 伸缩量 γ。
-  **variance_epsilon**: 小偏移量，防止除 0.
-  **name**: 操作名称，可选。

#### Returns:

归一化后的张量