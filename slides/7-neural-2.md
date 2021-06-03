---
title:  'Neural networks'
author: 'Fraida Fund'
---


\newpage

## Recap

* Last week: neural networks
* Many parameters
* Train using gradient descent
* How to compute gradients?

## Backpropagation

### How to compute gradients?

* Gradient descent requires computation of the gradient $\nabla L(\theta)$
* Backpropagation is key to efficient computation of gradients

### Composite functions and computation graphs

Suppose we have a composite function $f(g(h(x)))$

We can represent it as a computational graph, where each connection is an input and each node performs a function or operation:

\begin{tikzpicture}
  \node[circle] (n1) at (1,1) {$x$};
  \node[circle,fill=blue!20] (n2) at (4,1)  {$v=h(x)$};
  \node[circle,fill=blue!20] (n3) at (8,1)  {$u=g(v)$};
  \node[circle,fill=blue!20] (n4) at (11,1) {$f(u)$};
  \draw [->] (n1) -- (n2) node[midway, above] {};
  \draw [->] (n2) -- (n3) node[midway, above] {};
  \draw [->] (n3) -- (n4) node[midway, above] {};
\end{tikzpicture}


### Forward pass on computational graph

To compute the output $f(g(h(x)))$, we do a *forward pass* on the computational graph:

* Compute $v=h(x)$
* Compute $u=g(v)$
* Compute $f(u)$

### Derivative of composite function

* Suppose need to compute the derivative of the composite function $f(g(h(x)))$ with respect to $x$  

* We will use the chain rule.

### Backward pass on computational graph

We can compute this chain rule derivative by doing a *backward pass* on the computational graph:

We just need to get the derivative of each node with respect to its inputs: 

$$\frac{df}{dx} = \frac{df}{du} \frac{dg}{dv} \frac{dh}{dx}$$


\begin{tikzpicture}
  \node[circle] (n1) at (1,1) {$x$};
  \node[circle,fill=blue!20] (n2) at (4,1)  {$v=h(x)$};
  \node[circle,fill=blue!20] (n3) at (8,1)  {$u=g(v)$};
  \node[circle,fill=blue!20] (n4) at (11,1) {$f(u)$};
  \draw [<-] (n1) -- (n2) node[midway, above] {$\frac{dh}{dx}$};
  \draw [<-] (n2) -- (n3) node[midway, above] {$\frac{dg}{dv}$};
  \draw [<-] (n3) -- (n4) node[midway, above] {$\frac{df}{du}$};
\end{tikzpicture}


### Neural network computation graph



\begin{tikzpicture}
  \node[circle, fill=green!20] (n1) at (1,1) {$x_i$};
  \node[circle,fill=purple!20] (n2) at (4,1)  {$z_{H,i}$};
  \node[circle,fill=purple!20] (n3) at (8,1)  {$u_{H,i}$};
  \node[circle,fill=purple!20] (n4) at (11,1) {$z_{O,i}$};
  \node[circle,fill=orange!20] (n5) at (15,1) {$L(y_i, z_{O,i})$};
  \node[circle,fill=blue!20] (n6) at (1,4) {$W_H, b_H$};
  \node[circle,fill=blue!20] (n7) at (8,4) {$W_O, b_O$};


  \draw [->] (n1) -- (n2) node[midway, above] {};
  \draw [->] (n2) -- (n3) node[midway, above] {};
  \draw [->] (n3) -- (n4) node[midway, above] {};
  \draw [->] (n4) -- (n5) node[midway, above] {};
  \draw [->] (n6) -- (n2) node[midway, above] {};
  \draw [->] (n7) -- (n4) node[midway, above] {};

\end{tikzpicture}



### Backpropagation error: illustration

\begin{tikzpicture}
  \node[circle, fill=purple!20,minimum size=1cm] (n1) at (1,1) {};
  \node[circle,fill=purple!20,minimum size=1cm] (n2) at (1,5)  {};
  \node[label=$j$,circle,fill=purple!20,minimum size=1cm] (n3) at (5,3)  {};

  \draw [->] (n1) -- (n3) node[midway, above] {};
  \draw [->] (n2) -- (n3) node[midway, above] {$ u_i w_{j,i}$};
\end{tikzpicture}

At a node $j$,

* $z_j = \sum_i w_{j,i} u_{i}$
* $u_j = g(z_j)$

### Backpropagation error: definition

* Chain rule: $\frac{\partial L}{\partial w_{j,i}} = \frac{\partial L}{\partial z_{j}} \frac{\partial z_{j}}{\partial w_{j,i}}$

* Denote the backpropagation error of the node $j$ as $\delta_j = \frac{\partial L}{\partial z_j}$

* Since $z_j = \sum_i w_{j,i} u_{i}$, $\frac{\partial z_j}{\partial w_{j,i}} = u_i$

* Then $\frac{\partial L}{\partial w_{j,i}} = \delta_j u_i$


### Backpropagation error: output unit

For output unit in regression network, with

$$L =  \frac{1}{2}\sum_n (y_n - z_{O,n})^2$$

Then $\delta_O =  \frac{\partial L}{\partial z_O} = -(y_n - z_{O})$


### Backpropagation error: hidden unit illustration



\begin{tikzpicture}
  \node[circle, fill=purple!20,minimum size=1cm] (n1) at (1,1) {};
  \node[label=$i$,circle,fill=purple!20,minimum size=1cm] (n2) at (1,5)  {};
  \node[label=$j$,circle,fill=purple!20,minimum size=1cm] (n3) at (5,3)  {};
  \node[circle,fill=purple!20,minimum size=1cm] (n4) at (9,1)  {};
  \node[label=$k$,circle,fill=purple!20,minimum size=1cm] (n5) at (9,5)  {};

  \draw [->] (n1) -- (n3) node[midway, above] {};
  \draw [->] (n2) -- (n3) node[midway, above] {$ u_i w_{j,i}$};
  \draw [->] (n3) -- (n4) node[midway, above] {};
  \draw [->] (n3) -- (n5) node[midway, above] {$ u_j w_{k,j}$};
\end{tikzpicture}

### Backpropagation error: hidden unit

For a hidden unit,

$$\delta_j = \frac{\partial L}{\partial z_j} = \sum_k \frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial z_j}$$

$$\delta_j = \sum_k \delta_k \frac{\partial z_k}{\partial z_j} =  \sum_k \delta_k w_{k,j} g'(z_j) = g'(z_j) \sum_k \delta_k w_{k,j} $$

using $\delta_k = \frac{\partial L}{\partial z_k}$. And because $z_k = \sum_l w_{k,l} u_l = \sum_l w_{k,l} g(z_l)$, $\frac{\partial z_k}{\partial z_j}  = w_{k,j}g'(z_j)$.

\newpage

### Derivatives for common loss functions

Squared/L2 loss: 

$$L = \sum_i (y_i - z_{O,i})^2, \quad \frac{\partial L}{\partial z_{O,i}} = \sum_i -(y_i - z_{O,i})$$

Binary cross entropy loss: 

$$L = \sum_i -y_i z_{O,i} + \text{ln} (1 + e^{y_i z_{O,i}}), \quad \frac{\partial L}{\partial z_{O,i}} = y_i - \frac{ e^{y_i z_{O,i}} }{1 + e^{y_i z_{O,i}} }$$

### Derivatives for common activation functions

* Sigmoid activation: $g'(x) = \sigma(x) (1-\sigma(x))$
* Tanh activation: $g'(x) = \frac{1}{\text{cosh}^2(x)}$

### Backpropagation + gradient descent algorithm

1. Start with random (small) weights. Apply input $x_n$ to network and propagate values forward using $z_j = \sum_i w_{j,i} u_i$ and $u_j = g(z_j)$. (Sum is over all inputs to node $j$.)
2. Evaluate $\delta_k$ for all output units.
3. Backpropagate the $\delta$s to get $\delta_j$ for each hidden unit. (Sum is over all outputs of node $j$.)

$$\delta_j = g'(z_j) \sum_k w_{k,j} \delta_k$$

4. Use $\frac{\partial L_n}{\partial w_{j,i}} = \delta_j u_i$ to evaluate derivatives.
5. Update weights using gradient descent.


### Backpropagation demo notebook

[Link to demo notebook](https://colab.research.google.com/drive/1RXYHjJQiG97nzGiMNyJmHO9LxMQm9Sjh)

\newpage

### Why backpropagation?

![Forward-mode differentiation from input to output gives us derivative of every node with respect to each input. Then we can compute the derivative of output with respect to input. Image via [https://colah.github.io/posts/2015-08-Backprop/](https://colah.github.io/posts/2015-08-Backprop).](images/tree-forwradmode.png){width=60%}

![Reverse-mode differentiation from output to input gives us derivative of output with respect to every node. Image via [https://colah.github.io/posts/2015-08-Backprop/](https://colah.github.io/posts/2015-08-Backprop).](images/tree-backprop.png){width=60%}

\newpage

## Training challenges

Some models may not "converge" - why?
 
### Learning rate

* If learning rate is too high, weights can oscillate
* Can use adaptive learning rate algorithm like: momentum, RMSProp, Adam

### Local minima

* Error surface may have local minima
* Gradient descent can get "trapped"
* "Noise" can help get out of local minima: using stochastic gradient descent with one sample at a time, adding noise to data or weights, etc.


### Unstable gradients

* Backprop in a neural network involves multiplication of terms of the form $w_j g'(z_j)$
* When this term tends to be small: gradients get smaller and smaller as we move from output to input
* When this term tends to be large: gradients get larger and larger as we move from output to input

### Vanishing gradient problem: illustration

![Note the "flat spot" on the sigmoid, where the derivative is close to zero. Ideally, we want to operate in "linear" region of activation function](images/sigmoid-derivative.png){ width=40% }


### "Herd effect"

* Hidden units all move in the same direction at once, instead of "specializing"
* Solution: use initial (small!) random weights
* Use small initial learning rate so that hidden units can find "specialization" before taking large steps

### Many factors affect training efficiency

* Number of layers/hidden units
* Choice of activation function
* Choice of loss function

Classic paper: ["Efficient BackProp", Yann LeCun et al, 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

\newpage

## Training a neural network in Python

### Keras

* High-level Python library for building and fitting neural networks
* Runs on top of a *backend* 

### Backends for deep learning

Keras-compatible backends:

* TensorFlow (Google)
* CNTK (Microsoft)
* Theano (LISA Lab at Université de Montréal)


### PyTorch

* Also a high-level Python library for neural networks
* Developed by Facebook


### Keras recipe

1. Describe model architecture 
2. Select optimizer
3. Select loss function, compile model
4. Fit model
5. Test/use model


### Demo notebook

[Link to demo notebook (by Sundeep Rangan)](https://colab.research.google.com/github/sdrangan/introml/blob/master/unit09_neural/demo1_synthetic.ipynb)


\newpage


## Networks with multiple hidden layers


### Networks of linear units

* Recall: hidden layer doesn't do anything with linear activation function
* Equivalent to a single layer of weights

### Non-linear units and one layer of weights

![One layer of weights with non-linear activation creates a separating hyperplane.](images/nn-1-layer.png){ width=60%}

[TensorFlow Playground Link: Logistic regression on linearly separable data](https://playground.tensorflow.org/#activation=sigmoid&batchSize=30&dataset=gauss&regDataset=reg-gauss&learningRate=0.03&regularizationRate=0&noise=15&networkShape=&seed=0.29600&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


### Non-linear units and two layers of weights

![Two layers of weights with non-linear activation create a convex polygon region.](images/nn-2-layer.png){ width=60%}

[TensorFlow Playground Link: One hidden layer on circles](https://playground.tensorflow.org/#activation=sigmoid&batchSize=30&dataset=circle&regDataset=reg-gauss&learningRate=0.03&regularizationRate=0&noise=15&networkShape=3&seed=0.84765&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

\newpage

### Non-linear units and many layers of weights

![Three layers of weights with non-linear activation create a composition of polygon regions.](images/nn-3-layer.png){ width=60%}


### Deep networks

Networks with many hidden layers are challenging - 

* Computationally expensive to train
* Many parameters - at risk of overfitting
* Vanishing gradient problem

### Breakthroughs

In early 2010s, some breakthroughts

* Efficient training with GPU
* Huge data sets
* ReLu activation function 

