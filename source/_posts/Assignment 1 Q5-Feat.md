---
title: CS231 Assignment 1 Q5-Feat
date: 2020-01-12 09:25:00
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
### 1. 训练SVM分类器（多个学习率和正则化强度的准确率）    
**实现思路:**   
* 多次循环计算即可
 
```python
################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

svm = LinearSVM()
for learning_rate in learning_rates:
    for reg in regularization_strengths:
        loss_hist = svm.train(X_train_feats, y_train, learning_rate=learning_rate, reg=reg,
                      num_iters=3000, verbose=False)
        y_train_pred = svm.predict(X_train_feats)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred =svm.predict(X_val_feats)
        val_accuracy = np.mean(y_val == y_val_pred)
        results[(learning_rate,reg)] = (train_accuracy,val_accuracy)
        if val_accuracy>best_val:
            best_val = val_accuracy
            best_svm = svm

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

```

### 2.训练双层神经网络（参数自定）
* 实现思路：多个超参数多次循环进行训练，并计算val精度   

```python
################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

learning_rates = [1e-2,1e-1]
regs = [5e-6]
num_iter = [1000,2000]
best_val = 0

for num_iters in num_iter:
    for learning_rate in learning_rates:
        for reg in regs:
            net.train(X_train_feats,y_train,X_val_feats,y_val,learning_rate=learning_rate,
                learning_rate_decay=0.95,reg=reg,num_iters=num_iters,batch_size=200,verbose=True)
            y_val_pred = net.predict(X_val_feats)
            val_acc = np.mean(y_val == y_val_pred)
            if val_acc > best_val:
                best_val = val_acc
                best_net = net
                print('Temp best validation accuracy:',val_acc,'\t','best hyper param: ',[num_iter,learning_rate,reg])


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

 
## Summary
本次作业主要是体会特征改变带来的变化，本身难度并不高，与之前作业部分一致。  
 
## Reference
* [CS231n课程笔记翻译：神经网络笔记1（上）](https://zhuanlan.zhihu.com/p/21462488)