---
title: CS231 Assignment 2 Q1-FullyConnectedNet
date: 2020-01-13 09:25:00
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
### 1. affine layer forward & backward
**实现思路:**    
* 前向传播直接进行点积即可，需注意`reshape`  
* 反向传播依据求导公式计算即可
 
```python
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
 
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
 
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
 
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    X = x.reshape(x.shape[0],-1)
    out = X.dot(w) + b
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache
 
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
 
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)
 
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    x_reshape = x.reshape(x.shape[0],-1)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    dw = x_reshape.T.dot(dout)
    db = np.sum(dout,axis = 0)
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
```
 
 
### 2. ReLU layer forward & backward
**实现思路:**    
* 前向传播通过索引令`x<0`部分为0
* 反向传播依据求导公式计算即可，即大于0导数为1
 
```python
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
 
    Input:
    - x: Inputs, of any shape
 
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    out = x.copy()
    out[x < 0] = 0
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache
 
 
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
 
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
 
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    dx = (x > 0) * dout
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
```
 
### 3. 完成 TwoLayerNet 类
**实现思路:**   
* 权重初始化按照要求服从相应高斯分布，通过`np.random.normal()`函数生成
* 前向传播通过`layer_utils.py`整合的层进行
* 反向传播通过组合多个`backward`函数，需注意加入正则化
 
```python
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
 
    The architecure should be affine - relu - affine - softmax.
 
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
 
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
 
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
 
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
 
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim,num_classes))
        self.params['b2'] = np.zeros(num_classes)
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
 
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
 
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
 
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
 
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        a1, a1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        a2, a2_cache = affine_forward(a1, self.params['W2'], self.params['b2'])
        scores = a2
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
 
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        da2, grads['W2'], grads['b2'] = affine_backward(dscores,a2_cache)
        da1, grads['W1'], grads['b1'] = affine_relu_backward(da2,a1_cache)
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        return loss, grads
```
 
### 4. 训练一个双层神经网络  
**实现思路:**  
通过`TwoLayerNet`类构造神经网络，`Solver`调节超参数
```python
model = TwoLayerNet()
solver = None
 
##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# 50% accuracy on the validation set.                                        #
##############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
model = TwoLayerNet(reg=0.1)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                    'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
solver.train()
scores = model.loss(data['X_test'])
y_pred = np.argmax(scores, axis = 1)
acc = np.mean(y_pred == data['y_test'])
print('test acc: %f'%acc)
 
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
```
 
 
### 5. 完成 FullyConnectedNet 类  
**实现思路：**
* 通过循环来进行前向传播
* batchnorm normalization 和 layer normalization 初始化 gamma=1, beta=0
* 权重还是按照高斯分布初始化
* 反向传播仍然通过循环进行，多个层的反向传播函数进行组合
* 注：`affine_bn_relu_forward` `affine_bn_relu_backward` `affine_ln_relu_forward`  
    `affine_ln_relu_backward` 四个函数位于`layer_utils.py`文件，笔者自行加入
 
```python
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be
 
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
 
    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
 
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """
 
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
 
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
 
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        layer_input_dim = input_dim
        for i, hid in enumerate(hidden_dims):
          self.params['W%d'%(i+1)] = np.random.normal(loc=0.0, scale=weight_scale, size=(layer_input_dim,hid))
          self.params['b%d'%(i+1)] = np.zeros(hid)
          if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
            self.params['gamma%d'%(i+1)] = np.ones(hid)
            self.params['beta%d'%(i+1)] = np.zeros(hid)
          layer_input_dim = hid
        self.params['W%d'%(self.num_layers)] = np.random.normal(loc=0.0, scale=weight_scale, size=(layer_input_dim,num_classes))
        self.params['b%d'%(self.num_layers)] = np.zeros(num_classes)
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
 
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]
 
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
 
 
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
 
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
 
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        layer_input = X
        ar_cache = {}
        dp_cache = {}
        for i in range(self.num_layers-1):
          if self.normalization == None:
            layer_input, ar_cache[i] = affine_relu_forward(layer_input,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)])        
          else:
            if self.normalization == 'batchnorm':
              layer_input, ar_cache[i] = affine_bn_relu_forward(layer_input,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)],
                                                                self.params['gamma%d'%(i+1)], self.params['beta%d'%(i+1)], self.bn_params[i])
 
            elif self.normalization == 'layernorm':
              layer_input, ar_cache[i] = affine_ln_relu_forward(layer_input,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)],
                                                                self.params['gamma%d'%(i+1)], self.params['beta%d'%(i+1)], self.bn_params[i])
 
          if self.use_dropout:
            layer_input, dp_cache[i] = dropout_forward(layer_input,self.dropout_param)
 
        ar_out, ar_cache[self.num_layers-1] = affine_forward(layer_input,self.params['W%d'%self.num_layers],self.params['b%d'%self.num_layers])
        scores = ar_out   
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        # If test mode return early
        if mode == 'test':
            return scores
 
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W%d'%self.num_layers] * self.params['W%d'%self.num_layers])
        dx, grads['W%d'%self.num_layers], grads['b%d'%self.num_layers] = affine_backward(dscores, ar_cache[self.num_layers-1])
        grads['W%d'%self.num_layers] += self.reg * self.params['W%d'%self.num_layers]
        for i in range(self.num_layers-1,0,-1):
          if self.use_dropout:
            dx = dropout_backward(dx, dp_cache[i-1])
 
          if self.normalization == None:
            dx, grads['W%d'%i], grads['b%d'%i] = affine_relu_backward(dx, ar_cache[i-1])
          else:
            if self.normalization == 'batchnorm':
              dx, grads['W%d'%i], grads['b%d'%i], grads['gamma%d'%i], grads['beta%d'%i] = affine_bn_relu_backward(dx, ar_cache[i-1])
            if self.normalization == 'layernorm':
              dx, grads['W%d'%i], grads['b%d'%i], grads['gamma%d'%i], grads['beta%d'%i] = affine_ln_relu_backward(dx, ar_cache[i-1])
 
          grads['W%d'%i] += self.reg * self.params['W%d'%i]
          loss += 0.5 * self.reg * np.sum(self.params['W%d'%i] * self.params['W%d'%i])
 
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        return loss, grads
```
 
### 6. SGD with momentum
**实现思路：**  
* 按照公式计算
 
```python
def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
 
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
 
    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v
 
    return next_w, config
```
 
### 8. RMSProp & Adam 算法
**实现思路：**  
* 按照公式计算
 
```python
def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
 
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
 
    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw ** 2
    next_w = w - config['learning_rate'] * dw / (np.sqrt(config['cache'] + config['epsilon']))
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
 
    return next_w, config
 
 
def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
 
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)
 
    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw ** 2)
    m_unbias = config['m'] / (1 - config['beta1'] ** config['t'])
    v_unbias = config['v'] / (1 - config['beta2'] ** config['t'])
    next_w = w - config['learning_rate'] * m_unbias / (np.sqrt(v_unbias) + config['epsilon'])
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
 
    return next_w, config
```
 
### 9. 训练一个模型
```python
best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# find batch/layer normalization and dropout useful. Store your best model in  #
# the best_model variable.                                                     #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
learning_rate = 3.1e-4
weight_scale = 2.5e-2 #1e-5
model = FullyConnectedNet([600, 500, 400, 300, 200, 100],
                weight_scale=weight_scale, dtype=np.float64, dropout=0.5, normalization='batchnorm', reg=1e-2)
solver = Solver(model, data,
                print_every=50, num_epochs=10, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=0.9
         )
 
solver.train()
best_model = model
 
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
```
 
## Summary
本次作业亲手搭建一个神经网络，编写全连接层、ReLU层前向和反向传播，以及组合层。个人做起来还是比较吃力，  
在实现任意层神经网络时BN层和LN层卡了很久，其实应该在实现BN层和LN层再来写相关代码，会顺畅很多。
 
## Reference
* [CS231n课程笔记翻译：神经网络笔记3（下）](https://zhuanlan.zhihu.com/p/21798784)
* [lightaime/cs231n](https://github.com/lightaime/cs231n)