---
layout:     post   				    
title:      Deep learning with python 				 
subtitle:   summary of book -Deep learning with python
date:       2019-11-10 				
author:     Little Fat Tiger					 
header-img: img/post-bg-2015.jpg 	 
catalog: true 						 
tags:								 
    - deep learning
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
3.5 Here introduce a miltiple calss classification problem. An example with 46 labels. End with softmax activation function.

3.6 Here is a regression problem with prediction of house price. Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.
```python
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                            input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
#The network ends with a single unit and no activation (it will be a linear layer)
#Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the hidden layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.
```

# Chapter 4

4.1 Different type of machine learning. Supervised learning, Unsupervised learning, Self-supervised learning, Reinforcement learning. I think it should only be 3 not 4.

4.2 It talks about the model performance evaluation. It explains what is overfitting and information leak. Every time you tune a hyperparameter of your model based on the model’s performance on the validation set, some information about the validation data leaks into the model.

It introduces several way of separating data: SIMPLE HOLD-OUT VALIDATION； K-FOLD VALIDATION； ITERATED K-FOLD VALIDATION WITH SHUFFLING

Trains the final model on all non- test data available

4.3 Data preprocessing, feature engineering, and feature learning.

4.4 It discusses about the overfitting and underfitting. Nothing special here. To prevent a model from learning misleading or irrelevant patterns found in the training data, the best solution is to get more training data. Second to best: The processing of fighting overfitting this way is called regularization.  **L1 regularization—The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights). L2 regularization—The cost added is proportional to the square of the value of the weight coefficients**  Also mention ***dropout*** technique. Only dropout in training and adjust the weight on testing.

>Dropout, applied to a layer, consists of randomly dropping out (setting to zero) a number of output features of the layer during training. Let’s say a given layer would normally return a vector [0.2, 0.5, 1.3, 0.8, 1.1] for a given input sample during training. After applying dropout, this vector will have a few zero entries distributed at random: for example, [0, 0.5, 1.3, 0, 1.1]. The dropout rate is the fraction of the features that are zeroed out; it’s usually set between 0.2 and 0.5. At test time, no units are dropped out; instead, the layer’s output values are scaled down by a factor equal to the dropout rate, to balance for the fact that more units are active than at training time.

In sum 4 ways to reduct overfitting:
>Get more training data. Reduce the capacity of the network. Add weight regularization. Add dropout.

4.5 General workflow of ML. Metric: For balanced-classification problems, where every class is equally likely, accuracy and area under the receiver operating characteristic curve (ROC AUC) are common metrics. For class-imbalanced problems, you can use precision and recall. 

**I think chapter 1-4 are for foundation knowledge and chapters 5-9 are for practical example. 

# Chapter 5 Apply on computer vision. About convolutional neural networks.

```python
from keras import layers 
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```

```python
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
```
5.1 Here it introduce some basic concept and important parameters.  Border effects; strides; max-pooling operation.

5.2 
> Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pre- trained model—will constitute your future toolbox for tackling the problem of per- forming image classification with small datasets.

5.5-5.3 These two part mainly talk about how to apply CNN on the cat and dog classification. In 5.2, it uses a simple CNN from scratch to build a model. The performance is not very good. On 5.3, it uses a pretrain model on the top to help us. There are two ways to use the pretrain model. 

> Running the convolutional base over your dataset, recording its output to a Numpy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in part 1 of this book. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.

>Extending the model you have (conv_base) by adding Dense layers on top, and running the whole thing end to end on the input data. This will allow you to use data augmentation, because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.

The first one is fast but weeker and can not use any data augment. Second one is slower but more accurate. The first one only use the output of the pretrain model to serve as the input for final dense layer, it is very fast. Second method can do any fine tuning on the pretrain model. Usually we only fine turn the later layer because the initial layer has some basic information such as edge and point, which is useful for us and do not need to train again. The second model can reach 97% accuracy.

5.4 Visualizing for CNN. Visualizing intermediate activations; Visualizing convnet filters; Visualizing heatmaps of class activation

>  The first layer acts as a collection of various edge detectors. At that stage, the activations retain almost all of the information present in the initial picture.
As you go higher, the activations become increasingly abstract and less visually
interpretable. They begin to encode higher-level concepts such as “cat ear” and “cat eye.” Higher presentations carry increasingly less information about the visual contents of the image, and increasingly more information related to the class of the image.
The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image; but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image

```diff
-how to understand filter?
```

filter->pooling->dropout.

# Chapter 6 Deep learning for text and sequences- by using recurrent neural networks and 1D convnets

Applications: 
```
> 1 Document classification and timeseries classification, such as identifying the topic of an article or the author of a book
> 2 Timeseries comparisons, such as estimating how closely related two docu- ments or two stock tickers are
> 3 Sequence-to-sequence learning, such as decoding an English sentence into French
> 4 Sentiment analysis, such as classifying the sentiment of tweets or movie reviews as positive or negative
> 5 Timeseries forecasting, such as predicting the future weather at a certain loca- tion, given recent weather data
```

This chapter include : 

> This chapter’s examples focus on two narrow tasks: sentiment analysis on the IMDB dataset, a task we approached earlier in the book, and temperature forecasting.

6.1 Vectorizing text is the process of transforming text into numeric tensors. This can be done in multiple ways: Segment text into words, and transform each word into a vector; Segment text into characters, and transform each character into a vector;Extract n-grams of words or characters, and transform each n-gram into a vector.N-grams are overlapping groups of multiple consecutive words or characters. Collectively, the different units into which you can break down text (words, charac- ters, or n-grams) are called tokens, and breaking text into such tokens is called tokeniza- tion.

Here use a simple example to show how can we use text to sentiment classification.

6.2 RNN. 

> In effect, an RNN is a type ofneural network that has an internal loop. y = activation(dot(state_t, U) + dot(input_t, W) + b)

First it has simple RNN. Later it include LSTM and GRU.  Just keep in mind what the LSTM cell is meant to do: allow past informa- tion to be reinjected at a later time, thus fighting the vanishing-gradient problem.

6.3 3 kinds of RNN. Recurrent dropout; Stacking recurrent layers; Bidirectional recurrent layers. Stack means increase the layer of model. This could be done when the model did not have overfitting problem. Overfitting can be overcome by dropout method.

6.4 Sequence processing with convnets. We can also combine CNN and RNN together. 

Wrapping up:

> In the same way that 2D convnets perform well for processing visual patterns in 2D space, 1D convnets perform well for processing temporal patterns. They offer a faster alternative to RNNs on some problems, in particular natural- language processing tasks.

> Typically, 1D convnets are structured much like their 2D equivalents from the world of computer vision: they consist of stacks of Conv1D layers and Max- Pooling1D layers, ending in a global pooling operation or flattening operation.

> Because RNNs are extremely expensive for processing very long sequences, but 1D convnets are cheap, it can be a good idea to use a 1D convnet as a prepro- cessing step before an RNN, shortening the sequence and extracting useful rep- resentations for the RNN to process.


# Chapter 7

7.1 **functional API** These three important use cases—multi-input models, multi-output models, and graph-like models—aren’t possible when using only the Sequential model class in Keras.  --INCEPTION MODULES； -- RESIDUAL CONNECTIONS

7.2 ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.

What we learn: 
> 1. Keras callbacks provide a simple way to monitor models during training and automatically take action based on the state of the model.
> 2. When you’re using TensorFlow, TensorBoard is a great way to visualize model activity in your browser. You can use it in Keras models via the TensorBoard call- back.

7.3 Here it introduce some advanced technique. Like ensemble, hyperparameter optimization.

# Chapter 8

8.1 Use RNN to build a generated text model. Sampling the next token requires balance between adhering to what the model judges likely, and introducing randomness. One way to handle this is the notion of softmax temperature. Always experi- ment with different temperatures to find the right one.

8.2 Talking about DeepDream picture building. Combine several picture together and build a new picture by using CNN.

8.3 From Wrapping up
```
1 Style transfer consists of creating a new image that preserves the contents of a target image while also capturing the style of a reference image.
2 Content can be captured by the high-level activations of a convnet.
3 Style can be captured by the internal correlations of the activations of different
layers of a convnet.
4 Hence, deep learning allows style transfer to be formulated as an optimization
process using a loss defined with a pretrained convnet.
5 Starting from this basic idea, many variants and refinements are possible.
```

8.4 is VAE and 8.5 is GAN.

8.5 
```
1 A GAN consists of a generator network coupled with a discriminator network. The discriminator is trained to differenciate between the output of the generator and real images from a training dataset, and the generator is trained to fool the discriminator. Remarkably, the generator nevers sees images from the training set directly; the information it has about the data comes from the discriminator.
2 GANs are difficult to train, because training a GAN is a dynamic process rather than a simple gradient descent process with a fixed loss landscape. Getting a GAN to train correctly requires using a number of heuristic tricks, as well as extensive tuning.
3 GANs can potentially produce highly realistic images. But unlike VAEs, the latent space they learn doesn’t have a neat continuous structure and thus may not be suited for certain practical applications, such as image editing via latent- space concept vectors.
```
# Chapter 9
```
**This chapter covers**
1 Important takeaways from this book
2 The limitations of deep learning
3 The future of deep learning, machine learning, and AI
4 Resources for learning further and working in the field
```
