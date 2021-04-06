---
title: CS231 Assignment 2 Q4-ConvNet
date: 2020-01-16 09:25:00
mathjax: true
categories: CS231 Assignment
tags:
  - Deep Learning
  - Assignment
  - CS231n
---

## Introduction
本文为斯坦福大学CS231n课程作业及总结，若有错误，欢迎指正。  
所有代码均已上传到GitHub项目[cs231n-assignment2](https://github.com/notplus/cs231n-assignment/tree/master/assignment2)
 
## Code
### 1. Convolution: Naive forward  
**实现思路:** 通过循环计算卷积层前向传播，效率较低
 
```python
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
 
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
 
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
 
 
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
 
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_new = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N,F,H_new,W_new))
 
    for i in range(H_new):
      for j in range(W_new):
        x_masked = x_new[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]        
        for c in range(F):
          out[:,c,i,j] = np.sum(x_masked * w[c,:,:,:],axis=(1,2,3)) + b[c]
 
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache
 
```
 
### 2. convolutional backward   
**实现思路：** 求导计算梯度，进行反向传播      
 
```python
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
 
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
 
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout,axis=(0,2,3))
    dx_pad = np.zeros_like(x_pad)
 
    for i in range(H_new):
      for j in range(W_new):
        x_masked = x_pad[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]        
        for c in range(F):
          dw[c ,: ,: ,:] += np.sum(x_masked * (dout[:, c, i, j])[:, None, None, None], axis=0)
        for n in range(N):
          dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += np.sum(w[:,:,:,:]*(dout[n,:,i,j])[:,None,None,None],axis=(0))
 
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
 
```
 
### 3. Max-Pooling: Naive forward   
**实现思路：** 简单方式计算Maxpooling前向传播      
 
```python
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
 
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
 
    No padding is necessary here. Output size is given by
 
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = np.zeros((N,C,H_out,W_out))
 
    for i in range(H_out):
      for j in range(W_out):
        x_masked = x[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
        out[:,:,i,j] = np.max(x_masked,axis=(2,3))
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache
 
```
 
### 4. Max-Pooling: Naive backward   
**实现思路：** 简单方式计算Maxpooling反向传播      
 
```python
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
 
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
 
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    x, pool_param = cache
    dx = np.zeros_like(x)
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
 
    for i in range(H_out):
      for j in range(W_out):
        x_masked = x[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
        max_mask = np.max(x_masked,axis=(2,3))
        temp_mask = x_masked == (max_mask)[:,:,None,None]
        dx[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width] += temp_mask * (dout[:,:,i,j])[:,:,None,None]
 
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
 
```
 
### 5. Max-Pooling: Naive backward   
**实现思路：** 简单方式计算Maxpooling反向传播      
 
```python
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
 
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
 
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    x, pool_param = cache
    dx = np.zeros_like(x)
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
 
    for i in range(H_out):
      for j in range(W_out):
        x_masked = x[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
        max_mask = np.max(x_masked,axis=(2,3))
        temp_mask = x_masked == (max_mask)[:,:,None,None]
        dx[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width] += temp_mask * (dout[:,:,i,j])[:,:,None,None]
 
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
 
```

### 6 简单三层卷积神经网络
**实现思路：**
* 初始化和之前相同，权重采用高斯分布，偏差使用零初始化
* 前向传播和反向传播与之前三层全连接层实现类似，都是通过已经写好的前向/反向传播组合计算

```python
class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(scale=weight_scale,size=(num_filters,C,filter_size,filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(int(H/2)*int(W/2)*num_filters,hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim,num_classes))
        self.params['b3'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        a1, a1_cache = conv_relu_pool_forward(X,self.params['W1'],self.params['b1'],conv_param,pool_param)
        a2, a2_cache = affine_relu_forward(a1, self.params['W2'], self.params['b2'])
        a3, a3_cache = affine_forward(a2, self.params['W3'],self.params['b3'])
        scores = a3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))
        dx, grads['W3'], grads['b3'] = affine_backward(dscores,a3_cache)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, a2_cache)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, a1_cache)
        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
```

### 7. spatial batch normalization forward && backward
**实现思路：**
* 参考论文 [1] ，batch normalization 在卷积层的实现，不同于全连接的实现，由于要保留原有的空间结构且减少计算量，故将每一个channel作为feature进行归一化    
* 由于与之前实现的全连接bn层十分类似，可以直接调用之前的函数，只需对输入输出进行reshape   

```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (N,D) (D,)
    N, C, H, W = x.shape
    temp_out, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W,C)),gamma,beta,bn_param)
    out = temp_out.reshape(N, W, H, C).transpose(0,3,2,1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape(N*H*W,C), cache)
    dx = dx.reshape(N,W,H,C).transpose(0,3,2,1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

```

### 8. spatial groupnorm forward && backward
**实现思路：**
* 参考论文[2] 分组进行归一化   

![20200207211202.png](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200207211202.png)

```python
def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_trans = x.reshape(N, G, C // G, H, W)
    sample_mean = np.mean(x_trans, axis=(2,3,4), keepdims=True)
    sample_var = np.var(x_trans, axis=(2,3,4), keepdims=True)
    x_hat = (x_trans - sample_mean) / np.sqrt(sample_var + eps)
    x_hat = x_hat.reshape(N, C, H, W)
    out = gamma * x_hat + beta
    cache = (x, gamma, sample_mean, sample_var, eps, x_hat, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, gamma, mean, var, eps, x_hat, G = cache
    dgamma = np.sum(dout * x_hat, axis=(0,2,3), keepdims=True)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
    N, C, H, W = x.shape
    x_trans = x.reshape(N, G, C // G, H, W)
    m = C // G * H * W
    dx_hat = (dout * gamma).reshape(N, G, C // G, H, W)
    dvar = np.sum(dx_hat * (x_trans - mean) * (-0.5) * np.power((var + eps), -1.5), axis=(2,3,4), keepdims=True)
    dmean = np.sum(dx_hat * (-1) / np.sqrt(var + eps), axis=(2,3,4), keepdims=True) + dvar * np.sum(-2 * (x_trans - mean), axis=(2,3,4), keepdims=True) / m
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x_trans - mean) / m + dmean / m
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
```


## Summary
本次作业主要是有关卷积神经网络的一系列实现，包括卷积层、池化层、BN层的前反向传播实现，对于Group BN 还是没弄清楚。
 
## Reference
* [1] [Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015.](https://arxiv.org/abs/1502.03167)
* [2] [Wu, Yuxin, and Kaiming He. "Group Normalization." arXiv preprint arXiv:1803.08494 (2018).](https://arxiv.org/abs/1803.08494)
* [CS231n课程笔记翻译：神经网络笔记 2](https://zhuanlan.zhihu.com/p/21560667)
* [Group Normalization](https://blog.csdn.net/mooneve/article/details/83858012)
* [lightaime/cs231n](https://github.com/lightaime/cs231n)
 