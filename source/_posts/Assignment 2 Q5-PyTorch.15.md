---
title: CS231 Assignment 2 Q5-PyTorch
date: 2020-01-17 09:25:00
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
### 1. 实现一个三层全连接神经网络
**实现思路:** 通过PyTorch实现一个三层神经网络 使用全连接层  根据官方文档进行组合即可 
 
```python
def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    channel_1, _, KH1, KW1 = conv_w1.shape
    channel_2, _, KH2, KW2 = conv_w2.shape
    conv_1 = F.conv2d(x, conv_w1, bias=conv_b1, padding=2)
    relu_1 = F.relu(conv_1)
    conv_2 = F.conv2d(relu_1, conv_w2, bias=conv_b2, padding=1)
    relu_2 = F.relu(conv_2)
    faltten_1 = flatten(relu_2)
    fc_1 = faltten_1.mm(fc_w)
    scores = fc_1 + fc_b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores
```
 
### 2. 参数初始化     
**实现思路：** 调用函数进行参数初始化          
 
```python

################################################################################
# TODO: Initialize the parameters of a three-layer ConvNet.                    #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

conv_w1 = random_weight((32,3,5,5))
conv_b1 = zero_weight(32)
conv_w2 = random_weight((16,32,3,3))
conv_b2 = zero_weight(16)
fc_w = random_weight((16384,10))
fc_b = zero_weight(10)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)
 
```
 
### 3. 实现一个三层卷积神经网络   
**实现思路：** 根据官方文档实现即可        
 
```python
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv_1 = nn.Conv2d(in_channel, channel_1, (5,5), padding=2)
        nn.init.kaiming_normal_(self.conv_1.weight)
        #nn.init.kaiming_normal_(self.conv1.bias)
        self.conv_2 = nn.Conv2d(channel_1, channel_2, (3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_2.weight)
        #nn.init.kaiming_normal_(self.conv_2.bias)
        self.fc = nn.Linear(channel_2*32*32,num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                          END OF YOUR CODE                            #       
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = flatten(x)
        x = self.fc(x)
        scores = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores
 
```
 
### 4. 通过序列API实现三层神经网络
**实现思路：** 搭建网络很简单，参数的初始化如果按照题目要求的使用`random_weight`函数，结果很差     
 
```python

################################################################################
# TODO: Rewrite the 2-layer ConvNet with bias from Part III with the           #
# Sequential API.                                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model = nn.Sequential(
    nn.Conv2d(3,channel_1,(5,5),padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1,channel_2,(3,3),padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2*32*32,10)
)
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,nesterov=True)

# def init_weights(m):
#     if type(m) == nn.Conv2d or type(m) == nn.Linear:
#         m.weight.data = random_weight(m.weight.size())
#         #m.weight.data = nn.init.kaiming_normal_(m.weight)
#         m.bias.data = zero_weight(m.bias.size())

# model.apply(init_weights)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             
################################################################################
 
```
 
### 5. 自行搭建网络对CIFAR-10进行分类
以下是我参考[resnet](http://zh.d2l.ai/chapter_convolutional-neural-networks/resnet.html)实现的Resnet-18

```python
################################################################################
# TODO:                                                                        #         
# Experiment with any architectures, optimizers, and hyperparameters.          #
# Achieve AT LEAST 70% accuracy on the *validation set* within 10 epochs.      #
#                                                                              #
# Note that you can use the check_accuracy function to evaluate on either      #
# the test set or the validation set, by passing either loader_test or         #
# loader_val as the second argument to check_accuracy. You should not touch    #
# the test set until you have finished your architecture and  hyperparameter   #
# tuning, and only run the test set once at the end to report a final value.   #
################################################################################
model = None
optimizer = None

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

class Residual(nn.Module):
    def __init__(self, in_channel, out_channels, use_1x1_conv=False, strides=1):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channel, out_channels, (3,3), padding=1, stride=strides)
        nn.init.kaiming_normal_(self.conv_1.weight)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_2.weight)
        if use_1x1_conv:
            self.conv_3 = nn.Conv2d(in_channel,out_channels,(1,1),stride=strides)
            nn.init.kaiming_normal_(self.conv_3.weight)
        else:
            self.conv_3 = None
        self.use_1x1_conv = use_1x1_conv
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        Y = F.relu(self.bn_1(self.conv_1(x)))
        Y = self.bn_2(self.conv_2(Y))
        if self.conv_3:
            x = self.conv_3(x)
        return F.relu(Y+x)

def resnet_block(nn_module, in_channels, out_channels, num_residuals, block_index, first_block=False):
    for i in range(num_residuals):
        if i == 0 and not first_block:
            stride = 1
            if block_index > 3:
                stride = 2
            nn_module.add_module('resnet_block_{:d}_{:d}'.format(block_index,i),Residual(in_channels, out_channels,use_1x1_conv=True,strides=stride))
        elif i == 0:
            nn_module.add_module('resnet_block_{:d}_{:d}'.format(block_index,i),Residual(in_channels, out_channels)) 
        else:
            nn_module.add_module('resnet_block_{:d}_{:d}'.format(block_index,i),Residual(out_channels, out_channels)) 

model = nn.Sequential(
    nn.Conv2d(3,64,(7,7),stride=1,padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3,3),stride=1,padding=1)
)

resnet_block(model,64,64,2,1,first_block=True)
resnet_block(model,64,128,2,2)
resnet_block(model,128,256,2,3)
resnet_block(model,256,512,2,4)
model.add_module('avgpool',nn.AdaptiveAvgPool2d((1,1)))
model.add_module('flatten',nn.Flatten())
model.add_module('linear',nn.Linear(512,10))

optimizer = optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,nesterov=True)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             
################################################################################

# You should get at least 70% accuracy
train_part34(model, optimizer, epochs=10)
```


## Summary
本次作业主要是关于PyTorch的使用，大概了解了PyTorch的用法，可以看一些网络的代码，其实大同小异。
 
## Reference
* [5.11. 残差网络（ResNet）](http://zh.d2l.ai/chapter_convolutional-neural-networks/resnet.html)
* [TORCH.NN](https://pytorch.org/docs/stable/nn.html#conv2d)
