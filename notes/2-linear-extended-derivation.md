---
title:  'Simple linear regression - extended derivation of OLS solution'
author: 'Fraida Fund'
---



## Set up

We assume a linear model

$$\hat{y_i} = w_0 + w_1 x_i$$


Given the (convex) loss function

$$ MSE(w_0, w_1) = \frac{1}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i) ]^2 $$

to find the minimum, we take the derivative and set it equal to zero:

$$ \frac{\partial MSE}{\partial w_0} = 0, \frac{\partial MSE}{\partial w_1} = 0$$


## Solution for intercept $w_0$

First, let's solve for the intercept $w_0$. Using the chain rule, power rule:

$$ 
\frac{\partial MSE}{\partial w_0} = \frac{1}{n}\sum_{i=1}^n  (2)[y_i - (w_0 + w_1 x_i) ] (-1) = -\frac{2}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i)] 
$$

(We can then drop the constant factor when we set this expression equal to $0$.)

Then, setting $\frac{\partial MSE}{\partial w_0}=0$  is equivalent to setting the sum of residuals to zero:

$$ \sum_{i=1}^n e_i  = 0$$

(where $e_i$ is the residual term for sample $i$). 


## Solution for slope $w_1$

Next, we work on the slope:


$$ \frac{\partial MSE}{\partial w_1} = \frac{1}{n}\sum_{i=1}^n  2[y_i - (w_0 + w_1 x_i)](-x_i)$$

$$  \implies -\frac{2}{n} \sum_{i=1}^n x_i [y_i - (w_0 + w_1 x_i)]  = 0$$


Again, we can drop the constant factor. Then, this is equivalent to:

$$  \sum_{i=1}^n x_i e_i  = 0$$

(where $e_i$ is the residual term for sample $i$).

## Solving two equations for two unknowns

From setting the $\frac{\partial MSE}{\partial w_0}=0$ and $\frac{\partial MSE}{\partial w_1}=0$ we end up with two equations involving the residuals:

$$  \sum_{i=1}^n e_i  = 0,  \sum_{i=1}^n x_i e_i  = 0$$

where 

$$ e_i = y_i - (w_0 + w_1 x_i) $$

We can expand $\sum_{i=1}^n e_i  = 0$ into

$$  \sum_{i=1}^n  y_i = n w_0 + \sum_{i=1}^n x_i w_1 $$

then divide by $n$, and we find the intercept

$$w_0 = \frac{1}{n} \sum_{i=1}^n y_i - w_1 \frac{1}{n} \sum_{i=1}^n x_i $$

i.e.

$$w_0^* = \bar{y} - w_1 \bar{x}$$

where $\bar{x}, \bar{y}$ are the sample means of $x, y$.

\newpage

To solve for $w_1$, expand $\sum_{i=1}^n x_i e_i  = 0$ into

$$  \sum_{i=1}^n x_i y_i = \sum_{i=1}^n  x_i w_0 + \sum_{i=1}^n x_i^2 w_1 $$

and multiply by $n$.



$$  n \sum_{i=1}^n x_i y_i = n \sum_{i=1}^n  x_i w_0 + n \sum_{i=1}^n x_i^2 w_1 $$


Also, multiply the "expanded" version of $\sum_{i=1}^n e_i  = 0$, 

$$  \sum_{i=1}^n  y_i = n w_0 + \sum_{i=1}^n x_i w_1 $$


by $\sum x_i$, to get

$$  \sum_{i=1}^n x_i \sum_{i=1}^n  y_i = n \sum_{i=1}^n x_i w_0 + (\sum_{i=1}^n x_i)^2 w_1 $$


Now, we can subtract to get

$$  n \sum_{i=1}^n x_i y_i - \sum_{i=1}^n x_i \sum_{i=1}^n  y_i = n \sum_{i=1}^n x_i^2 w_1  - (\sum_{i=1}^n x_i)^2 w_1 $$

$$ = w_1 \left( n \sum_{i=1}^n x_i^2 - (\sum_{i=1}^n x_i)^2 \right) $$


and solve for $w_1^*$:

$$ w_1^*  = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) }{\sum_{i=1}^n (x_i - \bar{x})^2}$$
