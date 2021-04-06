---
title: CS231 Assignment 1 Q1-KNN
date: 2020-01-07 09:25:00
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
### 1. 通过两层循环计算L2   
**实现思路:** 对train和test数据切片，对应进行L2计算


```python
    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.


        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.


        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                slice_train=self.X_train[j:j+1,:]
                slice_test=X[i:i+1,:]
                dists[i,j]=np.sqrt(np.sum(np.power(slice_train-slice_test,2)))


                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```




### 2. 实现`predict_labels` 函数   
**实现思路：** 
* 通过调用`npargsort`函数，实现对单层test数据L2的排序，输出结果是从小到大排序后的下标。
* 这句代码`closest_y = self.y_train[train_topK_index]`用到了整型数组访问语法，即取出`self.y_train`中以`train_topK_index`中包含的值为下标的内容。
* `y_pred[i]=Counter(closest_y).most_common(1)[0][0]`使用了Collections模块，函数返回一个TopN列表
* 也可使用`count = np.bincount(closest_y)`和`y_pred[i] = np.argmax(count)`来得到y_pred[i]，关于np.bincount可参考[np.bitcount](https://blog.csdn.net/xlinsist/article/details/51346523)
```python
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.


        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.


        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


            index_array = np.argsort(dists[i, :])
            train_topK_index = index_array[:k]
            closest_y = self.y_train[train_topK_index]
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


            y_pred[i]=Counter(closest_y).most_common(1)[0][0]
            
            #count = np.bincount(closest_y)
            #y_pred[i] = np.argmax(count)


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        return y_pred
```


### 3. 通过一层循环计算L2   
**实现思路:**   
直接对整个训练集图片操作，此时`self.X_train`的大小为5000×3072，而`X[i]`的大小为1×3072，两者相减会自动对`X[i]`进行广播，使其扩展到与`self.X_train`相同的大小。
```python
    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.


        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


            dists[i, :]=np.sqrt(np.sum(np.power(X[i]-self.X_train,2),axis=1))


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```


### 4. 不通过循环计算L2（矩阵与广播）
**实现思路:**    
* 对L2计算公式拆成两个平方项和交叉项，交叉项利用矩阵乘法，平方项通过`np.sum`函数计算，最后利用广播机制相加
* 对于`test_square=np.sum(X**2,axis=1,keepdims=True)`，需要`keepdims=True`，保证`test_square.shape`为(500,1)使可以进行广播，否则`test_square.shpae`为(500,)，进行广播时会出错。
```python
    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.


        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        cross=np.multiply(np.dot(X,self.X_train.T),-2)
        train_square=np.sum(self.X_train**2,axis=1)
        test_square=np.sum(X**2,axis=1,keepdims=True)
        dists=np.add(dists,cross)
        dists=np.add(dists,test_square)
        dists=np.add(dists,train_square)
        dists=np.sqrt(dists)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```


### 5. 交叉验证
**实现思路:**   
* 使用`np.array_split`函数分割训练集及标签
* 交叉验证原理参考：   
>交叉验证。有时候，训练集数量较小（因此验证集的数量更小），人们会使用一种被称为交叉验证的方法，这种方法更加复杂些。还是用刚才的例子，如果是交叉验证集，我们就不是取1000个图像，而是将训练集平均分成5份，其中4份用来训练，1份用来验证。然后我们循环着取其中4份来训练，其中1份来验证，最后取所有5次验证结果的平均值作为算法验证结果。   
* 使用`np.concatenate`函数进行数组拼接，关于该函数用法可参考[文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html),`X_train_folds[:n]+X_train_folds[n+1:]`结果为list，对axis=0拼接


```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]


X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

X_train_folds=np.array_split(X_train,num_folds)
y_train_folds=np.array_split(y_train,num_folds)


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


for k in k_choices:
    each_acc=[]
    for n in range(num_folds):
        X_train_unit=np.concatenate((X_train_folds[:n]+X_train_folds[n+1:]),axis=0)
        y_train_unit=np.concatenate((y_train_folds[:n]+y_train_folds[n+1:]),axis=0)
        classifier.train(X_train_unit,y_train_unit)
        dists=classifier.compute_distances_no_loops(X_train_folds[n])
        Yval_predict=classifier.predict_labels(dists,k=k)
        acc=np.mean(Yval_predict==y_train_folds[n])
        each_acc+=[acc]
    k_to_accuracies[k]=each_acc



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```


* 交叉验证的结果如图，accuracy在k=10附近达到最大   
![20200121132703.png](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200121132703.png)

## Summary
本次作业主要是KNN分类器的实现，由于对于python及numpy模块的不熟悉，导致作业完成得很吃力，希望能够加强掌握。   
这次作业也发现了做科学计算或者深度学习相关内容，处理数据时的效率问题，for循环的效率明显低于向量、矩阵运算（本次作业里的测试，for循环大概用时200-300s，向量、矩阵运算大概用时0.68s）尽管矩阵运算需要消耗大量内存资源，但还是时间宝贵嘛。当然，具体问题还需要具体分析。


## Reference
* [cs231n classification](http://cs231n.github.io/classification/)
* [CS231n课程笔记翻译：图像分类笔记（上）](https://zhuanlan.zhihu.com/p/20894041?refer=intelligentunit)
* [CS231n课程笔记翻译：图像分类笔记（下）](https://zhuanlan.zhihu.com/p/20900216)
* [CS231n课程笔记翻译：Python Numpy教程](https://zhuanlan.zhihu.com/p/20878530?refer=intelligentunit)
* [CS231n课程作业（一）KNN分类器](https://www.jianshu.com/p/275eda2294ea)