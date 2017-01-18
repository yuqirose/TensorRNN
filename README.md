# TensorCompress
Codebase for neural network model compression using tensors. We show that tensor can capture higher order interactions with much fewer parameters, thus significantly reduce the memory overhead. We demonstrate the advantages using three type of neural network layers

1. Fully-connected (dense) layer

2. Convolutional layer

3. Recurrent layer


## [tensornet](https://github.com/USC-Melady/TensorCompress/tree/master/tensornet)
Core tensor compression layer. Implemented with different tensor models

1. layers: tensor reformuation of neural network layer

  * CP\_dense.py: dense layer with CP compression

  * Tucker\_dense.py: dense layer with Tucker compression

  * Mf\_dense.py: dense layer with Matrix factorization

2. tt\_decomp: deprecated tensor train operations

## [experiments](https://github.com/USC-Melady/TensorCompress/tree/master/experiments)
Example applications demonstrating the tensor compression

1. [mnist](http://yann.lecun.com/exdb/mnist/): hand-written digit recognition, demo for dense layer

  * 2-layer-cp: 2-layer dense layer with CP compress model

2. [ptb](https://www.cis.upenn.edu/~treebank/): text annotation, demo for recurrent layer

  * LSTM network: higher order temporal correlation
    $$x(t)$$
3. [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html): image classification, demo for convolutional layer

4. [data](../experiments/data): raw as well as processed data





