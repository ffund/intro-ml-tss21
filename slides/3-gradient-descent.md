---
title:  'Gradient descent'
author: 'Fraida Fund'
---

::: {.cell .markdown}

:::notes

**Math prerequisites for this lecture**: You should know about:

* derivatives and optimization (Appendix C in Boyd and Vandenberghe)
* complexity of algorithms (Big O notation)

:::

:::


## In this lecture

* Runtime of OLS solution for multiple/LBF regression
* Solution using gradient descent 
* Variations on main idea

\newpage

## Runtime of OLS solution

### Limitations of OLS solution

* Specific to linear regression, L2 loss
* For extremely large datasets: runtime, memory

### Background: Big O notation

Approximate the number of operations required, as a function of input size.

* Ignore constant terms, constant factors
* Ignore all but the dominant term

Example: $3n^3 + 100n^2 + 1000$?


### Computing OLS solution

We had

$$\mathbf{w^*} = \left(\Phi^T \Phi \right)^{-1} \Phi^T \mathbf{y}$$

where $\Phi$ is an $n \times d$ matrix. If $n \geq d$ then it is (usually) full rank and a unique solution exists.

How long does it take to compute?

::: notes

Runtime of a "naive" solution using "standard" matrix multiplication: 

* $O(d^2n)$ to multiply $\Phi^T \Phi$
* $O(dn)$ to multiply $\Phi^T y$
* $O(d^3)$ to compute the inverse of $\Phi^T \Phi$ 

Since $n$ is generally much larger than $d$, the first term dominates and the runtime is $O(d^2n)$. 

(Note: in practice, we can do it a bit faster.)


:::

## Solution using gradient descent

### Exhaustive search


### Iterative solution

Suppose we would start with all-zero or random weights. Then iteratively (for $t$ rounds):

* pick random weights
* if loss performance is better, keep those weights
* if loss performance is worse, discard them

::: notes

For infinite $t$, we'd eventually find optimal weights - but clearly we could do better.

:::

\newpage


### Background: Gradients and optimization 

Gradient has *two* important properties for optimization:

At a minima (or maxima, or saddle point), 

$$\nabla L(\mathbf{w}) = 0$$

At other points, $\nabla L(\mathbf{w})$ points towards direction of maximum (infinitesimal) rate of *increase*.


### Gradient descent idea

To move towards minimum of a (smooth, convex) function, use first order approximation: 

Start from some initial point, then iteratively 

* compute gradient at current point, and 
* add some fraction of the negative gradient to the current point


### Gradient descent illustration

![[Link for animation](https://miro.medium.com/max/700/1*KQVi812_aERFRolz_5G3rA.gif). Image credit: Peter Roelants](../images/gradient-descent-animation.gif){width=80%}

<!--

### Visual example: least square solution 3D plot

![Regression parameters - 3D plot.](../images/3.2b.svg){ width=40% }

-->



### Standard ("batch") gradient descent

For each step $t$ along the error curve:

$$
\begin{aligned}
\mathbf{w}^{t+1} &= \mathbf{w}^t - \alpha \nabla L(\mathbf{w}^t) \\
 &= \mathbf{w}^t - \frac{\alpha}{n} \sum_{i=1}^n \nabla L_i(\mathbf{w}^t, \mathbf{x}_i, y_i)
\end{aligned}
$$


Repeat until stopping criterion is met.

\newpage


### Example: gradient descent for linear regression (1)

With a mean squared error loss function

$$ 
\begin{aligned}
L(w, X, y) &= \frac{1}{n} \sum_{i=1}^n (y_i - \langle w, x_i \rangle)^2 \\
     &= \frac{1}{n} \|y - Xw\|^2 
\end{aligned}
$$


### Example: gradient descent for linear regression (2)



we will compute the weights at each step as

$$
\begin{aligned} 
w^{t+1} &= w^t + \frac{\alpha^t}{n} \sum_{i=1}^n (y_i - \langle w^t,x_i \rangle) x_i \\
        &= w^t + \frac{\alpha^t}{n} X^T (y - X w^t)                  
\end{aligned}
$$

(dropping the constant 2 factor)

::: notes


To update $\mathbf{w}$, must compute $n$ loss functions and gradients - each iteration is $O(nd)$. We need multiple iterations, but in many cases it's more efficient than the previous approach.

However, if $n$ is large, it may still be expensive!

:::



## Variations on main idea

::: notes


Two main "knobs" to turn:

* "batch" size
* learning rate

:::


### Stochastic gradient descent 

Idea: 

At each step, compute estimate of gradient using only one randomly selected sample, and move in the direction it indicates.

Many of the steps will be in the wrong direction, but progress towards minimum occurs *on average*, as long as the steps are small.

::: notes

Each iteration is now only $O(d)$, but we may need more iterations than for gradient descent. However, in many cases we still come out ahead (especially if $n$ is large!).

See [supplementary notes](https://chinmayhegde.github.io/introml-notes-sp2020/pages/lecture3_notes.html) for an analysis of the number of iterations needed.

Also:

* SGD is often more efficient because of *redundancy* in the data - data points have some similarity.
* If the function we want to optimize does not have a global minimum, the noise can be helpful - we can "bounce" out of a local minimum.

:::

\newpage

### Mini-batch (also "stochastic") gradient descent (1)

Idea: 

At each step, select a small subset of training data ("mini-batch"), and evaluate gradient on that mini-batch. 

Then move in the direction it indicates.

### Mini-batch (also "stochastic") gradient descent (2)


For each step $t$ along the error curve: 

* Select random mini-batch $I_t\subset{1,\ldots,n}$
* Compute gradient approximation:

$$g^t = \frac{1}{|I_t|} \sum_{i\in I_t} \nabla L(\mathbf{x}_i, y_i, \mathbf{w}^t)$$

* Update parameters: $\mathbf{w}^{t+1} = \mathbf{w}^t - \alpha^t g^t$

::: notes

This approach is often used in practice because we get some benefit of vectorization, but also take advantage of redundancy in data.
:::



<!-- 

https://www.cs.cornell.edu/courses/cs4787/2021sp/
https://www.cs.cornell.edu/courses/cs6787/2018fa/Lecture2.pdf


https://ruder.io/optimizing-gradient-descent/

https://sebastianraschka.com/faq/docs/sgd-methods.html
https://sebastianraschka.com/faq/docs/gradient-optimization.html
https://distill.pub/2017/momentum/
https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L12_optim_slides.pdf
https://vis.ensmallen.org/
https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L05_gradient-descent_slides.pdf

-->



### Selecting the learning rate


![Choice of learning rate $\alpha$ is critical](../images/learning_rate_comparison.png){ width=85% }

::: notes

Also note: SGD "noise ball"

:::




### Annealing the learning rate

One approach: decay learning rate slowly over time, such as 

* Exponential decay: $\alpha_t = \alpha_0 e^{-k t}$
* 1/t decay: $\alpha_t = \alpha_0 / (1 + k t )$ 

(where $k$ is tuning parameter).


::: notes

But: this is still sensitive, requires careful selection of gradient descent parameters for the specific learning problem. 

Can we do this in a way that is somehow "tuned" to the shape of the loss function?

:::


\newpage

### Gradient descent in a ravine (1)

![Gradient descent path bounces along ridges of ravine, because surface curves much more steeply in direction of $w_1$.](../images/ravine-grad-descent.png){width=50%}

### Gradient descent in a ravine (2)

![Gradient descent path bounces along ridges of ravine, because surface curves much more steeply in direction of $w_1$.](../images/ravine-grad-descent2.png){width=50%}

### Momentum (1)

* Idea:  Update includes a *velocity* vector $v$, that accumulates gradient of past steps. 
* Each update is a linear combination of the gradient and the previous updates. 
* (Go faster if gradient keeps pointing in the same direction!)

\newpage

### Momentum (2)

Classical momentum: for some $0 \leq \gamma_t < 1$,

$$v_{t+1} = \gamma_t v_t - \alpha_t \nabla L\left(w_t\right)$$

so

$$w_{t+1} = w_t + v_{t+1} = w_t  - \alpha_t \nabla L\left(w_t\right) + \gamma_t v_t$$

($\gamma_t$ is often around 0.9, or starts at 0.5 and anneals to 0.99 over many epochs.)

Note: $v_{t+1} = w_{t+1} - w_t$ is $\Delta w$.

### Momentum: illustrated

![Momentum dampens oscillations by reinforcing the component along $w_2$ while canceling out the components along $w_1$.](../images/ravine-momentum.png){width=50%}

### RMSProp

Idea: Track per-parameter EWMA of square of gradient, to normalize parameter update step. 


$$w_{t+1} = w_t  - \frac{\eta}{\sqrt{ v_{t+1} + \epsilon } } \nabla L\left(w_t\right) $$

where

$$v_{t} = \gamma v_{t-1} + (1 - \gamma) \nabla L\left(w_{t}\right) ^2$$


::: notes

Weights with recent gradients of large magnitude have smaller learning rate, weights with small recent gradients have larger learning rates.

:::
### Adam: Adaptive Moment Estimation

* Uses ideas from momentum (first moment) and RMSProp (second moment)!
* plus bias correction 
<!-- https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture04.pdf -->

::: notes

Bias correction accounts for the fact that first and second moment estimates start at zero.

:::

\newpage

### Illustration (Beale's function)


![Animation credit: Alec Radford. [Link for animation](https://imgur.com/a/Hqolp).](../images/beale-gradient.gif){width=40%}


::: notes

Due to the large initial gradient, velocity based techniques shoot off and bounce around, while those that scale gradients/step sizes like RMSProp proceed more like accelerated SGD.

:::

### Illustration (Long valley)


![Animation credit: Alec Radford. [Link for animation](https://imgur.com/a/Hqolp).](../images/long-valley-gradient.gif){width=40%}

::: notes

SGD stalls and momentum has oscillations until it builds up velocity in optimization direction. Algorithms that scale step size quickly break symmetry and descend in optimization direction.

:::


## Recap

* Gradient descent as a general approach to model training
* Variations

