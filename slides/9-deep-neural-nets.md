---
title:  'Deep learning'
author: 'Fraida Fund'
---

\newpage

## In this lecture

* Deep neural networks
* Challenges and tricks

:::notes

**Math prerequisites for this lesson**: None.

:::



## Recap

Last week: neural networks with one hidden layer


* Hidden layer learns feature representation
* Output layer learns classification/regression tasks


::: notes

With the neural network, the "transformed" feature representation is *learned* instead of specified 
by the designer.


![Image is based on a figure in Deep learning, by Goodfellow, Bengio, Courville. ](../images/8-deep-learning-motivation.png){ width=100% }

A neural network with non-linear activation, with one hidden layer and many units in that layer *can* approximate virtually any continuous real-valued function, with the right weights.  (Refer to the *Universal Approximation Theorem*.) But (1) it may need a very large number of units to represent the function, and (2) those weights might not be learned by gradient descent - the loss surface is very unfriendly.

Instead of a single hidden layer, if we use multiple hidden layers they can "compose" functions learned by the previous layers into more complex functions - use fewer units, and tends to learn better weights .


<!--

Universal approximation theorem: https://cedar.buffalo.edu/~srihari/CSE676/6.4%20ArchitectureDesign.pdf

-->

:::

\newpage

## Deep neural networks

::: notes

![Illustration of a deep network, with multiple hidden layers.](../images/8-deep-network.png){ width=60% }

Some comments:

* each layer is fully connected to the next layer
* each unit still works the same way: take the weighted sum of inputs, apply an activation function, and that's the unit output
* still trained by backpropagation

We call the number of layers the "depth" of the network and the number of hidden units in a layer its "width."


:::

### Double descent: animation

![Polynomial model before and after the interpolation threshold. Image source: [Boaz Barak, click link to see animation](https://windowsontheory.org/2021/01/31/a-blitz-through-classical-statistical-learning-theory/).](../images/8-polynomial-animation.gif){ width=40% }

::: notes

Explanation (via [Boaz Barak](https://windowsontheory.org/2021/01/31/a-blitz-through-classical-statistical-learning-theory/)):

> When $d$ of the model is less than $d_t$ of the polynomial, we are "under-fitting" and will not get good performance. As $d$ increases between $d_t$ and $n$, we fit more and more of the noise, until for $d=n$ we have a perfect interpolating polynomial that will have perfect training but very poor test performance. When $d$ grows beyond $n$, more than one polynomial can fit the data, and (under certain conditions) SGD will select the minimal norm one, which will make the interpolation smoother and smoother and actually result in better performance.



:::

\newpage

### Double descent curve


![Double descent curve (left) and realization in a real neural network (right).](../images/8-double-descent.png){ width=100% }


::: notes

With a deep neural network, we are not trying to operate in the "classical ML" bias-variance tradeoff regime (to the left of the interpolation threshold). 

(Interpolation threshold: where the model is just big enough to fit the training data exactly.)

- too-small models: can't represent the "true" function well (lacks capacity to learn complicated data representations!)
- too-big models (before interpolation threshold): memorizes the input, doesn't generalize well to unseen data (very sensitive to noise)
- REALLY big models: many possible weights that memorize the input, our challenge is to find the weight combination that memorizes the input *and* does well on unseen data

This is complicated by the "loss landscape" of a deep neural network (trained using backpropagation over the computational graph) looking not so friendly...

:::



### Loss landscape

!["Loss landscape" of a deep neural network in a "slice" of the high-dimensional feature space.](../images/resnet56_noshort_small.jpg){ width=30% }

::: notes

Image source: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

There are a variety of techniques we can use to improve the network's ability to learn good weights + find them efficiently, even on this difficult loss surface. (Including what we call "conditioning" techniques, and some broader techniques.)

:::

\newpage

### Addressing the challenges


::: notes

![Image credit: Sebastian Raschka](../images/deep-learning-markmap.svg){ width=60% }

:::

## Dataset

- Get more data
- **Data augmentation**
- **Use related data (transfer learning)**
- Use unlabeled data (self-supervised, semi-supervised)

::: notes

We won't talk much about getting more data, but if it's possible to get more labeled data, it is almost always the most helpful thing you can do!

* Example: JFT-300M, a Google internal dataset for training image models, has 300M images
* Example: GTP-3 trained on 45TB of compressed plaintext, about 570GB after filtering

*Reference: Revisiting Unreasonable Effectiveness of Data in Deep Learning Era. Chen Sun, Abhinav Shrivastava, Saurabh Singh, Abhinav Gupta; Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 843-852.*

It's not always possible to get a lot of *labeled* data for training a supervised learning model. But sometimes we can use *unlabeled* data, which is much easier to get. For example:

* In self-supervised learning, the label can be inferred automatically from the unlabeled data. e.g. GPT is trained to predict "next word".
* In semi-supervised/weakly-supervised learning, we generate labels (probably imperfectly) for unlabeled data (maybe using a smaller volume of labeled data to train a model to label the data!)

These are mostly out of scope of this course. But we *will* talk about data augmentation and transfer learning...

:::


\newpage


### Data augmentation

![Data augmentation on a cat image.](../images/cats-augmentation.png){ width=50% }


::: notes

It doesn't restrict network capacity - but it helps generalization by increasing the size of your training set! 

* For image data: apply rotation, crops, scales, change contrast, brightness, color.
* For text you can replace words with synonyms, simulate typos.
* For audio you can adjust pitch or speed, add background noise etc.

:::

### Transfer learning

Idea: leverage model trained on *related* data.

::: notes

State-of-the-art networks involve millions of parameters, huge datasets, and days of training on GPU clusters

But you don't have to repeat this process "from scratch" each time you train a neural network. (For many tasks, you may not even have enough data to get good results by training a network "from scratch".)

The "feature extraction" part of a neural network trained on related data, is likely still very useful for a slightly different task.

:::

### Transfer learning from pre-trained networks

Use pre-trained network for a different task

* Use early layers from pre-trained network, freeze their parameters
* Only train small number of parameters at the end

:::notes

The base model is a powerful feature extractor (learns transformation into a more useful feature space), then you just have to train for the mapping from this feature space to the target variable.

In practice: when applying deep learning, we almost always use a pre-trained base model! It saves time, energy, and cost. 


Example: To pre-train the 7B parameter (smallest) Llama 2, Meta's open source language model, takes 184,320 GPU hours (21 GPU YEARS!) and anywhere from $100,000-800,000 (depending on cost of GPU instances).

:::

\newpage

### Transfer learning illustration (1)

![When the network is trained on a very similar task, even the abstract high-level features are probably very relevant, so you might tune just the classification head.](../images/8-transfer-similar.png){ width=80% }

### Transfer learning illustration (2)

![If the original network is not as relevant, may fine-tune more layers.](../images/8-transfer-less.png){ width=80% }


\newpage

## Architecture/setup

- **Activation functions** (+ skip connections)
- **Weight initialization**
- **Convolutional units**
- Recurrent units
- ... many more ideas

:::notes

Again, this is mostly out of scope of this course, but we'll talk about some of these items very briefly. 

:::
### Recall: activation functions

![Candidate activation functions for a neural network.](../images/activation-functions.png){ width=40% }


<!--
### Zero-centered outputs

Remember that at each hidden unit, we compute

$\frac{\partial L_n}{\partial w_{j,i}} = \delta_j u_i$ 

where $\delta_j$ is the backpropagation error from the "upstream" nodes.

What happens if the $u_i$ terms are always positive?

::: notes

* To compute the derivative with respect to the weights at the input to a neuron, we compute the "local derivative" and then multiply by the "upstream" "backpropagation error" (scalar).

* The scalar multiplier may be positive or negative.

:::

-->


### Sigmoid: vanishing gradient (1)

![Computing gradients by backpropagation. At a hidden unit, $\delta_j = g'(z_j) \sum_k w_{k,j} \delta_k$. ](../images/deep-vanishing-network.png){ width=80% }

::: notes

Suppose we want to compute the gradient of the loss with respect to the weight shown in pink. 

* We will *multiply* local gradients (pink) all along each path between weight and loss function, starting from the end and moving toward the input. 
* Then we *add* up the products of all the paths. 
* With $\sigma$ activation, gradient along each path includes the product of a LOT of $\sigma '$ terms.

:::

\newpage

### Sigmoid: vanishing gradient (2)

![Sigmoid function and its derivative.](../images/deep-vanishing-sigmoid.png){ width=40% }

::: notes


What happens when you are in the far left or far right part of the sigmoid?

* Gradient is close to zero
* Weight updates are also close to zero
* The "downstream" gradients will also be values close to zero! (Because of backpropagation.)
* And, when you multiply quantities close to zero - they get even smaller.

The network "learns fastest" when the gradient is large. When the sigmoid "saturates", it "kills" the neuron!

Even the maximum value of the gradient is only 0.25 - so the gradient is always less than 1, and we know what happens if you multiply many quantities less than 1...


(tanh is slightly better - gradient has a larger max + some other advantages - still has vanishing gradient.)


:::


### ReLU: Dead ReLU


![ReLU function and its derivative.](../images/deep-dead-relu.png){ width=40% }


::: notes

ReLU is a much better non-linear function:

* does not saturate in positive region
* very very fast to compute

But, can "die" in the negative region. Once we end up there (e.g. by learning negative bias weight) we cannot recover - since gradient is also zero, gradient descent will not update the weights.

(ReLU is also more subject to "exploding" gradient than sigmoid/tanh.)

:::

\newpage

### ReLU: Leaky ReLU

![Leaky ReLU function and its derivative.](../images/deep-leaky-relu.png){ width=40% }

:::notes

When input is less than 0, the ReLU (and downstream units) is *completely* dead (not only very small!)

Alternative: **leaky ReLU** has small (non-zero) gradient $\alpha$ in the negative region - can recover. 

$$f(x) = \text{max}(\alpha x, x)$$

:::

<!-- 

### Other activations

![Other activations. Image via [Sefik Ilkin Serengil](https://sefiks.com/2020/02/02/dance-moves-of-deep-learning-activation-functions/).](../images/sample-activation-functions-square.png){ width=60% }


-->


### Skip connections

Alternative solution for "vanishing gradient":

* Direct connection between some higher layers and lower layers
* A "highway" for gradient info to go directly back to lower layers

::: notes


![Diagram of skip connection.](../images/deep-learning-skip.png)

Suppose we want this sequence of layers to learn $z = H(x)$.

Instead of learning $H(x)$ directly, the network learns the **residual function**:

$$\mathcal{F}(x) = H(x) - x$$

Then it reconstructs the desired transformation as:

$$H(x) = \mathcal{F}(x) + x$$

This helps avoid vanishing gradient.


In the standard neural network, we would have:

$$\frac{dL}{dx} = \frac{dL}{dz} \cdot g'(h(x)) \cdot h'(x)$$

where $g'()$ and $h'()$ multiply the gradient that backpropagates from the higher layers, $\frac{dL}{dz}$, and scale it down. But with the skip connection, we now have

$$\frac{dL}{dx} = \frac{dL}{dz} \cdot \left( \frac{d\mathcal{F}(x)}{dx} + I \right)
= \frac{dL}{dz} \cdot \left( g'(h(x)) \cdot h'(x) + I \right)$$

The identity matrix $I$ passes on the gradient from higher layers as an additive term, without scaling it down.

:::


### Weight initialization


![One path in forward pass on a neural network. ](../images/deep-network-initialization.png){ width=80% }

::: notes


What if we initialize weights to:

* **zero?** If weights are all initialized to zero, all the outputs are zero (for any input) - won't learn.
* **a constant (non-zero)?** If weights are all initialized to the same constant, we are more prone to "herding" - hidden units all move in the same direction at once, instead of "specializing".
* **a normal random value with small $\sigma$?** Small normal random values work well for "shallow" networks, but not for deep networks - it makes the outputs "collapse" toward zero at later layers.
* **a normal random value with large $\sigma$?** Large normal random values are bad - it makes the outputs "explode" at later layers.

:::

\newpage

### Weight initialization - normal

![Initial weights and ReLU unit outputs for each layer in a network.](../images/deep-weight-init-normal.png){ width=75% }


:::notes

* top row: too-small initial weights, by the last layer the outputs "collapse" toward zero
* middle row: good initial weights, distribution is similar from input to output
* bottom row: too-large initial weights, by the last layer the outputs "explode"

<!-- Image via [Andre Perunicic](https://intoli.com/blog/neural-network-initialization/) -->
:::


### Desirable properties for initial weights - principle

* The mean of the intial weights should be around 0 
* The variance of the activations should stay the same across every layer 

:::notes

If you are interested, [here's a derivation](https://www.deeplearning.ai/ai-notes/initialization/index.html).

:::

### Desirable properties for initial weights - practice

* For tanh: Xavier scales by $\frac{1}{\sqrt{N_{in}}}$
* For ReLU: He scales by $\frac{2}{\sqrt{N_{in}}}$

$N_{in}$ is the number of inputs to the layer ("fan-in").

### Weight initialization - He

![Initial weights and ReLU unit outpus for each layer in a network, He initialization. In this example, the size of the layers is: 100, 150, 200, 250, 300.](../images/deep-weight-init-he.png){ width=80% }


<!--

### Desirable properties - illustration (1)

![Activation function outputs with normal initialization of weights. Image source: Justin Johnson.](../images/8-init-collapse.png){ width=70% }



### Desirable properties - illustration (2)

![Activation function outputs with Xavier initialization of weights. Image source: Justin Johnson.](../images/8-init-xavier.png){ width=70% }

-->

\newpage

### Convolutional units

<!-- 

https://ajcr.net/stride-guide-part-2/

Explaining how it is implemented "flat" by matrix multiplication
https://ca.meron.dev/blog/Vectorized-CNN/

-->

### Problems with "fully connected" layers for images 


![Problems with fully connected layers.](../images/deep-image-data.svg){ width=60%}

::: notes

Convolutional neural networks address two major problems that make it difficult to train a "fully connected" neural network:

1. Each pixel is a feature, images tend to be very large, so a "fully connected" layer requires a very large number of parameters. (Each "neuron" in the first layer requires a weight for every pixel in the image!)
2. Each learned weight corresponds to specific pixels in the image. If the relevant pixels are at a different position in the image, the weights that were learned are not helpful for that other position. (i.e. it is not *spatial translation invariant*.)

::::

### Using convolution to address the problem 

![Addressing the problem.](../images/deep-image-conv.svg){ width=60%}

::: notes

With a convolutional layer (not "fully connected"):

1. The unit connects to one "patch" of the image at a time, so it only needs as many weights as there are pixels in the patch. The same weights are *shared* as the unit moves across the image, one patch at a time.
2. It can "match" a specific arrangement of pixels *anywhere* it occurs in the image, not only one specific location.

:::

\newpage


### The convolution operation - one "patch"

![Convolution on one patch.](../images/deep-image-conv-op.png){ width=80%}


::: notes

* Layer has a set of learnable "filters" (illustration shows one filter)
* Each filter has small width and height, but full depth
* During forward pass, filter "slides" across width and height of input, and computes dot product
* Effectively performs "convolution"

:::


### The convolution operation - stride and padding 

![Stride and padding.](../images/deep-conv-stride-pad.png){ width=80%}

\newpage

### The convolution operation - full depth

![Convolution across full depth.](../images/deep-image-conv-depth.png){ width=80%}



### The convolution operation - multiple filters

![Convolution with multiple filters.](../images/deep-conv-multi-filter.png){ width=80%}

::: notes

Note that the output is a 3D volume - can be input to another conv layer!

Basic dimension arithmetic:

* Accepts input volume $W_1 \times H_1 \times D_1$
* Four hyperparameters: number of filters $K$, filter size $F$, stride $S$, amount of zero padding $P$
* Produces volume of size 

$$W_2 = \left\lfloor \frac{W_1 - F + 2P}{S}\right\rfloor  + 1 , H_2 = \left\lfloor \frac{H_1 - F + 2P}{S} \right\rfloor + 1 $$
$$D_2 = K $$

* With parameter sharing: $F \cdot F \cdot D_1$ weights per filter, for $F \cdot F \cdot D_1 \cdot K$ weights and $K$ biases

:::

\newpage

### The convolution operation - all together

![Animated demo at [https://cs231n.github.io/assets/conv-demo/index.html](https://cs231n.github.io/assets/conv-demo/index.html)](../images/convolution-example.png){ width=90%}

Basic insight - parameter sharing:

* A particular filter with a set of weights represents a feature to look for
* If it is useful to look for a feature at position $x,y$, it is probably useful to look for the same feature at $x',y'$
* All neurons within a "depth slice" can share the same weights.




<!-- 

### Receptive field



Important for VGG -->

### Activation

* Convolutional typically followed by non-linear activation function e.g. ReLU
* Several Conv + ReLU layers may be followed by a *pooling* layer

\newpage

### Pooling layer

![Example of pooling layer.](../images/deep-conv-pool.png){ width=70% }


::: notes

* No parameters! Just applies an aggregating function
* Typically uses max operation (other possible operations: mean, median)
* Reduces spatial size of image (reduce computation, prevent overfitting)
* Typical example: 2x2 filter size, stride of 2, downsamples by a factor of 2 along width and height
* Works independently on each depth slice

Pooling math:

* Accepts input volume $W_1 \times H_1 \times D_1$
* Two hyperparameters: filter size $F$, stride $S$
* Produces volume of size 

$$W_2 = \left\lfloor \frac{W_1 - F}{S} \right\rfloor + 1 , H_2 = \left\lfloor \frac{H_1 - F}{S} \right\rfloor + 1 , D_2 = D_1 $$


:::

### The typical "LeNet"-like architecture 

!["LeNet"-like architecture.](../images/deep-conv-lenet-like.png){ width=80% }

\newpage

### Actual LeNet-5 (1998) 

![LeNet-5.](../images/deep-conv-lenet-5.png){ width=100% }


### Recurrent neural networks

* Where ConvNet find *spatial* patterns wherever they occur in images,
* RNNs find *temporal patterns* wherever they occur in a sequence



## Normalization 

* **Input standardization**
* **Batch normalization**

### Data pre-processing

You can make the loss surface much "nicer" by pre-processing:

* Remove mean (zero center)
* Normalize (divide by standard deviation)
* OR decorrelation (whitening/rotation)

::: notes

There are several reasons why this helps. We already discussed the "ravine" in the loss function that is created by correlated features. 

<!-- What about zero-centering and normalization? Think about a binary classification problem of a data cloud that is far from the origin, vs. one close to the origin. In which case will the loss function react more (be more sensitive) to a small change in weights?

![The classifier on the right is more sensitive to small changes in the weights.](../images/8-pre-processing.png){ width=30% }

-->

<!-- Note: Whitening/decorrelation is not applied to image data. For image data, we sometimes subtract the "mean image" or the per-color mean. -->
:::

### Data preprocessing (1)

![Image source: Stanford CS231n.](../images/8-preprocessing-1.jpeg){ width=50% }

\newpage

### Data preprocessing (2)

![Image source: Stanford CS231n.](../images/8-preprocessing-2.jpeg){ width=50% }



::: notes

Input standardization helps with the first hidden layer, but what about the intermediate hidden layers?

:::

### Batch normalization

* Re-center and re-scale between layers
* Training: Mean and standard deviation per training mini-batch
* Test: Using fixed statistics



## Gradient descent

<!-- 
### Standard ("batch") gradient descent

For each step $t$ along the error curve:

$$W^{t+1} = W^t - \alpha \nabla L(W^t) = W^t - \frac{\alpha}{N} \sum_{i=1}^N \nabla L_i(W^t, \mathbf{x}_i, y_i)$$


Repeat until stopping criterion is met.


### Stochastic gradient descent 

Idea: at each step, compute estimate of gradient using only one randomly selected sample, and move in the direction it indicates.

Many of the steps will be in the wrong direction, but progress towards minimum occurs *on average*, as long as the steps are small.

Bonus: helps escape local minima. 


### Mini-batch (also "stochastic") gradient descent

Idea: In each step, select a small subset of training data ("mini-batch"), and evaluate gradient on that mini-batch. Then move in the direction it indicates.

For each step $t$ along the error curve: 

* Select random mini-batch $I_t\subset{1,\ldots,N}$
* Compute gradient approximation: $g^t = \frac{1}{|I_t|} \sum_{i\in I_t} \nabla L(\mathbf{x}_i, y_i, W)$
* Update parameters: $W^{t+1} = W^t - \alpha^t g^t$


### Comparison: batch size

![Effect of batch size on gradient descent.](../images/grad-descent-comparison.png){ width=50% }


### Why does mini-batch gradient help? (Intuition)

* Standard error of mean over $m$ samples is $\frac{\sigma}{\sqrt{m}}$, where $\sigma$ is standard deviation.
* The benefit of more examples in reducing error is less than linear!
* Example: gradient based on 10,000 samples requires 100x more computation than one based on 100 samples, but reduces SE only 10x.
* Also: memory required scales with mini-batch size.
* Also: there is often redundancy in training set.


### Gradient descent terminology

* Mini-batch size is $B$, training size is $N$
* A training *epoch* is the sequence of updates over which we see all non-overlapping mini-batches
* There are $\frac{N}{B}$ steps per training epoch
* Data shuffling: at the beginning of each epoch, randomly shuffle training samples. Then, select mini-batches in order from shuffled samples.

\newpage

### Selecting the learning rate


![Choice of learning rate $\alpha$ is critical](../images/learning_rate_comparison.png){ width=55%}

### Annealing the learning rate

One approach: decay learning rate slowly over time, such as 

* Exponential decay: $\alpha_t = \alpha_0 e^{-k t}$
* 1/t decay: $\alpha_t = \alpha_0 / (1 + k t )$ 

(where $k$ is tuning parameter).


### Gradient descent in a ravine (1)

![Gradient descent path bounces along ridges of ravine, because surface curves much more steeply in direction of $w_1$.](../images/ravine-grad-descent.png){width=40%}

### Gradient descent in a ravine (2)

![Gradient descent path bounces along ridges of ravine, because surface curves much more steeply in direction of $w_1$.](../images/ravine-grad-descent2.png){width=40%}

\newpage

### Momentum (1)

* Idea:  Update includes a *velocity* vector $v$, that accumulates gradient of past steps. 
* Each update is a linear combination of the gradient and the previous updates. 
* (Go faster if gradient keeps pointing in the same direction!)


### Momentum (2)

Classical momentum: for some $0 \leq \gamma_t < 1$,

$$v_{t+1} = \gamma_t v_t - \alpha_t \nabla L\left(W_t\right)$$

so

$$W_{t+1} = W_t + v_{t+1} = W_t  - \alpha_t \nabla L\left(W_t\right) + \gamma_t v_t$$

($\gamma_t$ is often around 0.9, or starts at 0.5 and anneals to 0.99 over many epochs.)


### Momentum: illustrated

![Momentum dampens oscillations by reinforcing the component along $w_2$ while canceling out the components along $w_1$.](../images/ravine-momentum.png){width=50%}

### RMSProp

Idea: Track *per-parameter* EWMA of *square* of gradient, and use it to adapt learning rate. 


$$W_{t+1,i} = W_{t,i} -\frac{\alpha}
{\sqrt {\epsilon + E[g^2]_t }} \nabla L(W_{t,i})$$

where 

$$E[g^2]_t=(1-\gamma)g^2 + \gamma E[g^2]_{t-1}, \quad g = \nabla J(W_{t,i})$$


::: notes

Weights with recent gradients of large magnitude have smaller learning rate, weights with small recent gradients have larger learning rates.


:::

\newpage

### RMSProp: illustrated (Beale's function)


![Animation credit: Alec Radford. [Link to view animation](https://imgur.com/a/Hqolp).](../images/beale-gradient.gif){width=40%}


::: notes

Due to the large initial gradient, velocity based techniques shoot off and bounce around, RMSProps proceed more like faster SGD.

:::

### RMSProp: illustrated (Long valley)


![Animation credit: Alec Radford. [Link to view animation](https://imgur.com/a/Hqolp). ](../images/long-valley-gradient.gif){width=40%}

::: notes

SGD stalls and momentum has oscillations until it builds up velocity in optimization direction. Algorithms that scale step size quickly break symmetry and descend in optimization direction.

:::

### Adam: Adaptive moments estimation (2014)
 
Idea: Track the EWMA of *both* first and second moments of the gradient, $\{m_t, v_t\}$ at each time $t$. 

If $L_t(W)$ is evaluation of loss function on a mini-batch of data at time $t$, 

$$
\begin{aligned}
\{m_t, v_t\}, \mathbb{E}[m_t] \, \, &\approx \, \, \mathbb{E}[\,\nabla \, L_t(W)\,], \mathbb{E}[v_t] \,  \\
                                  \, &\approx \, \, \mathbb{E}\big[\,(\nabla \, L_t(W))^2\,\big]
 \end{aligned}
 $$

Scale $\alpha$ by $\frac{m_t}{\sqrt{v_t}}$ at each step.

\newpage
-->

## Regularization

- **L2 or L1 regularization**
- **Early stopping**
- **Dropout**



### L1 or L2 regularization

:::notes


As with other models, we can add a penalty on the norm of the weights.

Normal gradient descent update rule:

$$w_{i,j}^{t+1} = w_{i,j}^t - \alpha \frac{\partial L}{\partial w_{i,j}^t} $$


With L2 regularization: 

$$w_{i,j}^{t+1} = w_{i,j}^t - \alpha ( \frac{\partial L}{\partial w_{i,j}^t} + \frac{2 \lambda}{n}  w_{i,j}^t )$$



Often called "weight decay" in the context of neural nets.

:::

### Early stopping 


::: notes

* Compute validation loss each performance
* Stop training when validation loss hasn't improved in a while
* Risk of stopping *too* early


Important: *must* divide data into training, validation, and test data - use validation data (not test data!) to decide when to stop training.


<!-- 

Why does it work? Some ideas:

* The network is effectively "smaller" when we stop training early, because many units still in linear region of activation.
* Earlier layers (which learn simpler features) and late layers (near the output - used for response mapping) converge to their final weights first. See [Boaz Barak](https://windowsontheory.org/2021/02/17/what-do-deep-networks-learn-and-when-do-they-learn-it/).

-->


<!-- See "Why early stopping works" https://www.cs.toronto.edu/~guerzhoy/411/lec/W05/overfitting_prevent.pdf -->

<!-- See https://windowsontheory.org/2021/02/17/what-do-deep-networks-learn-and-when-do-they-learn-it/ -->

:::

### Dropout 

![Dropout networks.](../images/8-dropout.jpeg){ width=35% }

::: notes

* During each training step: some portion of neurons are randomly "dropped".
* During each test step: don't "drop" any neurons, but we need to scale activations by dropout probability


Why does it work? Some ideas:

* Forces some redundancy, makes neurons learn robust representation
* Effectively training an ensemble of networks (with shared weights)


Note: when you use Dropout layers, you may notice that the validation/test loss seems better than the training loss! Why?

:::



<!--

Neural networks of all types: https://www.asimovinstitute.org/neural-network-zoo/

-->




::: notes

---

Big picture: What are all these techniques for?

* Allow us to train models with more 'capacity'
* Improve performance even without adding 'capacity'

:::

## Example: Deep Neural Nets: 33 years ago and 33 years from now

![Techniques we will apply in this example.](../images/deep-nets-markmap-example.png){ width=50% }

:::notes

In the Colab lesson, we will reproduce a 1989 paper about a neural network for handwritten digit classification, that was used in the late 90s to process 10-20% of all checks in the US.

* The original paper had 5% error, our realization has about 4.14%
* With a bunch of these changes (but keeping the basic network the same) we get to 2.09%
* The original model without any changes, but with more training data, gets to 3.05%
* With our changes + more data, we get to 1.31%

What does it mean that we can do this without changing the basic network? It means the network always had the *capacity* to do this well, but wasn't learning the best weights.


:::

## Example: Progress on ImageNet

* Data: 1.2M images from 1000 categories
* ImageNet Large Scale Visual Recognition Challenge (ILSVRC): running since 2010

::: notes

![Progress on ImageNet.](../images/deep-conv-imagenet.png){ width=80%}

If you want to learn more, including receptive field, 1x1 convolution, and ConvNet architectures, I recommend:

* [Lecture 2A](https://fullstackdeeplearning.com/spring2021/lecture-2a/) and [Lecture 2B](https://fullstackdeeplearning.com/spring2021/lecture-2b/) from FSDL 2021
* [Chapter 7](https://d2l.ai/chapter_convolutional-neural-networks/index.html) and [Chapter 8](https://d2l.ai/chapter_convolutional-modern/index.html) of Dive into Deep Learning

:::
