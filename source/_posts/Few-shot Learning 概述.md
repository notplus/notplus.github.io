---
title: Few-shot Learning 概述
date: 2020-03-07 09:25:00
categories: Deep Learing
tags:
  - Deep Learning
  - Few-shot Learning
---

# Few-shot Learning 概述 
 
## What is few-shot Learning?
Few-shot Learning 是 Meta Learning 在监督学习领域的应用。 
Meta Learning，又称为learning to learn，在meta training阶段将数据集分解为不同的meta task，去学习类别变化的情况下模型的泛化能力，在meta testing阶段，面对全新的类别，不需要变动已有的模型，就可以完成分类。
 
few-shot的训练集中包含了很多的类别，每个类别中有多个样本。
在训练阶段，会在训练集中随机抽取C个类别，每个类别K个样本（总共CK个数据），构建一个 meta-task，作为模型的支撑集（support set）输入；
再从这C个类中剩余的数据中抽取一批（batch）样本作为模型的预测对象（batch set）。
即要求模型从C*K个数据中学会如何区分这C个类别，这样的任务被称为C-way K-shot(C类，每个类K个样本)问题。
训练过程中，每次训练 (episode)都会采样得到不同meta-task。这种机制使得模型学会不同meta-task中的共性部分，比如如何提取重要特征及比较样本相似等，忘掉 meta-task中task相关部分，因此，在面对新的未见过的 meta-task时，也能较好地进行分类。[1]
 
## Models 
### Optimization based
1. MAML [2]  
基本思路是通过学习模型的初始化参数使得其在一步或多步迭代下实现精度最大化。
![MAML](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200210185652.png)[3] 
 
1. Reptile [4]  
基本思路与MAML一致，一样是学习初始化参数，区别在于MAML训练时只一次迭代，取第二次迭代参数梯度更新；Reptile训练时多次迭代，参数向最后迭代方向作为更新方向。
![Reptile](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200210183327.png)[5] 
 
3. Gradient Descent as LSTM [6]
通过LSTM，将梯度下降看作LSTM的一种特例，通过学习forget gate & input gate 参数(正则化强度 & 学习率)，以期得到一种比梯度下降更好的优化算法。
![Gradient Descent as LSTM](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200210185441.png)[7] 
 
 
### Metric based  
1. Siamese neural networks [8] 
输入两张图片,一张训练,一张为测试,通过CNN提取特征,再经过embedding,比较相似度,计算分数.
 
2. Prototypical Network [9]  
寻找类别在高维空间的投影,以期通过聚类方式解决分类问题.  
![Prototypical Network](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200210192535.png)[10]   
 
3. Relation Network [11]  
relation network在计算相似度时通过一个cnn来学习计算.  
![Relation Network](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200210193006.png)[11]   
 
 
### Model based
1. Memory-Augmented Neural Networks[12]   
![Model based](https://raw.githubusercontent.com/wuliutx/upload-pic/master/20200210194232.png)[13]
 
 
## Reference
* [1] [few-shot learning](https://daiwk.github.io/posts/ml-few-shot-learning.html)
* [2] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
* [3] [Meta Learning – MAML (5/9)](https://youtu.be/vUwOA3SNb_E)
* [4] [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)
* [5] [Meta Learning – MAML (9/9)](https://youtu.be/9jJe2AD35P8)
* [6] [Optimization as a Model for Few-Shot Learning](https://openreview.net/pdf?id=rJY0-Kcll)
* [7] [Meta Learning - Gradient Descent as LSTM (2/3)](https://youtu.be/G_xYYq772NQ)
* [8] [Siamese neural networks for one-shot image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
* [9] [Prototypical networks for few-shot learning](https://arxiv.org/abs/1703.05175)
* [10] [Meta Learning – Metric-based (3/3)](https://youtu.be/semSxPP2Yzg)
* [11] [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)
* [12] [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/abs/1605.06065)
* [13] [Meta-Learning(2)---Memory based方法](https://zhuanlan.zhihu.com/p/61037404)