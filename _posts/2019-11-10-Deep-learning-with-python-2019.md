---
layout:     post   				    
title:      Deep learning with python 				 
subtitle:   summary of book -Deep learning with python
date:       2019-11-10 				
author:     Little Fat Tiger					 
header-img: img/post-bg-2015.jpg 	 
catalog: true 						 
tags:								 
    - book reading summary
---

It is the book Deep learning with python, author FRANÇOIS CHOLLET.

# Chapter 1
   1.1. Deep Learning is used for supervised learning. Initially itintroduce _loss function_, _Backpropagation_ for basic concept. The target for training is to minimize loss function. Talk about the history and development of AI.

   1.2. Brifely talk about the machine learning model before DL.1- Naive Bayes is a type of machine-learning classifier based on applying Bayes’ theorem while assuming that the features in the input data are all independent (a strong, or “naive” assumption, which is where the name comes from). 2- Logistic regression is for classification problem but with name regression. 3- Early DL. Its development is from the invention of backpropagation. 4- kernel methods. SVM is most famous. Mapping data to high dimention and the decision boundary can be expressed as hyperplane. 5- Decision tree. Random forest and GBM. 6 The development of DL on ImageNet and some competition. DL can do automatical feature engineer. Also talked about its benefit.
   
   1.3 The reason why DL develop. It is due to hardware, data, algorithm and investment. 
   
   In general, besides the basic model, we only need to learn GBM and DL. On GBM, we need to learn XGBoost, and light GBM.

# Chapter 2
   2.1 Show a DL example in MNIST with less than 20 line code of python.
   
   2.2 Example of tensor. From one dimension to more than 3 dimensions. 3 important property of tensor are 1, number of axes; 2, shape; 3, data type. It also talk about data manipulation.
```python
import numpy as np
x = np.array(12)
x = np.array([12, 3, 6, 14])
#example of slicing
my_slice = train_images[10:100, :, :]
my_slice = train_images[10:100, 0:28, 0:28]
my_slice = train_images[:, 7:-7, 7:-7]
```
It also talk about batch axis or batch dimension. It also talks about different data and it dimension from 2D to 5D. 2D Vector data; 3D Timeseries data; 4D image - (samples, height, width, color_depth); 5D Video (samples, frames, height, width, color_depth) .
 
   2.3 It talks about the tensor operation: element addition and multiple; dot multiple, also their naive implementation;  Reshape.
```python
(dot(x, y)) if and only if x.shape[1] == y.shape[0]

```   
   2.4 It talks about training model and how to update weight. It includes derivatives and gradient, Stochastic gradient descent->Update based on random bunch of data. Some basic info.

# Chapter 3

# Chapter 4

# Chapter 5

# Chapter 6

# Chapter 7

# Chapter 8

# Chapter 9
