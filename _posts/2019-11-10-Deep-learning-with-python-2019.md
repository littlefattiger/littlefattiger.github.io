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

3.1 Begin the deep learning process. Initially, it introduces the idea like building lego bricks on model to do deep learning and later use an example to tell us the code step by step.

3.2-3 It is still an example of Kera. Nothing special for this chapter. It also recommend you to buy a GPU or use cloud.

3.4 A simple IMDB dataset example.
```python
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])    

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
          results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)   

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))
# overfitting
history_dict = history.history
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['accuracy']) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() 
acc_values = history_dict['accuracy'] 
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

# Chapter 4

# Chapter 5

# Chapter 6

# Chapter 7

# Chapter 8

# Chapter 9
