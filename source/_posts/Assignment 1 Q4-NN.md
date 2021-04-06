---
title: CS231 Assignment 1 Q4-NN
date: 2020-01-11 09:25:00
mathjax: true
categories: CS231 Assignment
tags:
  - Deep Learning
  - Assignment
  - CS231n
---

## Introduction
本文为斯坦福大学CS231n课程作业及总结，若有错误，欢迎指正。   
所有代码均已上传到GitHub项目[cs231n-assignment1](https://github.com/notplus/cs231n-assignment/tree/master/assignment1)

## Code
### 1. 通过计算score,Loss和梯度   
**实现思路:** 该两层神经网络,可通过下图简要理解,注意反向传播的过程,梯度=上游*当前   
![bp.jpg](https://upload-images.jianshu.io/upload_images/2301760-f876d8918d8ab61a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2)    
   
$当j=i时$:   
$\frac{\partial Loss}{\partial s_{j}}=-\frac{1}{a_{j}} \cdot a_{j} \cdot\left(1-a_{j}\right)=a_{i}-1$
       
$当j\neq i时$:   
$\frac{\partial Loss}{\partial s_{j}}=-\frac{1}{a_{j}} \cdot-a_{j} \cdot a_{i}=a_{i}$    
   
$\frac{\partial Loss}{\partial W_{2}}= h \cdot \frac{\partial loss}{\partial s}$      
  
```python
    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        h_output = np.maximum(0, X.dot(W1) + b1)
        scores = h_output.dot(W2) + b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        shift_scores = scores - np.max(scores,axis=1).reshape(-1,1)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)
        loss = -np.sum(np.log(softmax_output[range(N),list(y)]))
        loss = loss / N + 1 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dscores = softmax_output.copy()
        dscores[range(N),list(y)] -= 1
        dscores /= N
        grads['W2'] = (h_output.T).dot(dscores) + 2 * reg * W2
        grads['b2'] = np.sum(dscores,axis=0)

        dh = dscores.dot(W2.T)
        dh_ReLU = (h_output > 0) * dh
        grads['W1'] = X.T.dot(dh_ReLU) + 2 * reg * W1
        grads['b1'] = np.sum(dh_ReLU,axis=0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads
```


### 2. 完成train函数   
**实现思路:**   
通过np.random.choice随机选择batchsize大小样本用于计算loss和grad，更新权重
```python
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            mask = np.random.choice(num_train,size=batch_size,replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W1'] -= grads['W1'] * learning_rate
            self.params['W2'] -= grads['W2'] * learning_rate

            self.params['b1'] -= grads['b1'] * learning_rate
            self.params['b2'] -= grads['b2'] * learning_rate

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
```

### 3. 计算多个学习率和正则化强度的准确率
**实现思路:**    
* 多次循环计算即可

```python
best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

input_size = 32 * 32 * 3
hidden_sizes = [50,100,500,1000]
num_classes = 10


num_iterss = [1000,2000,3000]
learning_rates = [1e-4,1e-3]
regs = [0.25,0.5,1]

best_acc = -1
best_hyper_param = []

# Train the network
for hidden_size in hidden_sizes:
    net = TwoLayerNet(input_size, hidden_size, num_classes)
    for num_iters in num_iterss:
        for learning_rate in learning_rates:
            for reg in regs:
                stats = net.train(X_train, y_train, X_val, y_val,
                        num_iters=num_iters, batch_size=200,
                        learning_rate=learning_rate, learning_rate_decay=0.95,
                        reg=reg, verbose=False)
                
                # Predict on the validation set
                val_acc = (net.predict(X_val) == y_val).mean()
                #print('Validation accuracy:',val_acc,'\t',[hidden_size,num_iters,learning_rate,reg])
                if val_acc>best_acc:
                    best_acc = val_acc
                    best_net = net
                    best_hyper_param = [hidden_size,num_iters,learning_rate,reg]
                    print('Temp best validation accuracy:',val_acc,'\t','best hyper param: ',[hidden_size,num_iters,learning_rate,reg])

print('Validation accuracy:',best_acc)
print('Best hyper parm:',best_hyper_parm)

            
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```


## Summary
本次作业主要是简单两层神经网络的实现，主要难点在于计算梯度，重点理解反向传播（BP算法）。   

## Reference
* [CS231n课程笔记翻译：反向传播笔记](https://zhuanlan.zhihu.com/p/21407711)
* [lightaime/cs231n](https://github.com/lightaime/cs231n)