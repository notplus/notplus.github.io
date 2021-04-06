---
title: 蓝桥杯 2018省赛 C/C++ A T4
date: 2021-04-06 08:54:18
categories: Algorithm
tags:
	- Lanqiao
---

# 蓝桥杯 2018省赛 C/C++ A T4

## 题目

标题：第几个幸运数

到x星球旅行的游客都被发给一个整数，作为游客编号。
x星的国王有个怪癖，他只喜欢数字3,5和7。
国王规定，游客的编号如果只含有因子：3,5,7,就可以获得一份奖品。

我们来看前10个幸运数字是：
3 5 7 9 15 21 25 27 35 45
因而第11个幸运数字是：49

小明领到了一个幸运数字 59084709587505，他去领奖的时候，人家要求他准确地说出这是第几个幸运数字，否则领不到奖品。

请你帮小明计算一下，59084709587505是第几个幸运数字。

需要提交的是一个整数，请不要填写任何多余内容。



## 思路

* 暴力法：循环进行直至最大值。耗时几十个小时，不可行
* 生成法：依次生成幸运数，确定第几个。
  * 生成过程中会出现重复元素，如何去除？
  * 生成过程中元素并非有序，如何排序？



## 解决方案

使用C++ STL 数据类型`set` 有序且保证元素唯一



## 代码

```cpp
#include <iostream>
#include <set>

using namespace std;

#define MAX 59084709587505

int main()
{
	int a[3] = {3, 5, 7};
	long long t = 1;
	set<long long> s;
	while (true)
	{
		for (int i = 0; i < 3; i++)
		{
			long long tt = t * a[i];
			if (tt <= MAX)
				s.insert(tt);
		}
		t = *(s.upper_bound(t));
		if (t >= MAX)
			break;
	}
	cout << s.size() << endl;
	return 0;
}

```

