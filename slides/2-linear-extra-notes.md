---
title:  'Simple linear regression - extended derivation of OLS solution'
author: 'Fraida Fund'
---

### Gradient descent illustration

![[Link for animation](https://miro.medium.com/max/700/1*KQVi812_aERFRolz_5G3rA.gif). Image credit: Peter Roelants](../images/gradient-descent-animation.gif){width=70%}



### Minimizing MSE - simple linear regression (1)

Given 

$$ MSE(w_0, w_1) = \frac{1}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i) ]^2 $$

we take

$$ \frac{\partial MSE}{\partial w_0} = 0, \frac{\partial MSE}{\partial w_1} = 0$$


### Minimizing MSE - simple linear regression (2)

First, the intercept:

$$ MSE(w_0, w_1) = \frac{1}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i) ] ^2 $$

$$ \frac{\partial MSE}{\partial w_0} = \frac{1}{n}\sum_{i=1}^n  (2)[y_i - (w_0 + w_1 x_i) ] (-1)$$

$$  = -\frac{2}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i)] $$

using chain rule, power rule. 

::: notes

(We can then drop the $-2$ constant factor when we set this expression equal to $0$.)


:::

### Minimizing MSE - simple linear regression (3)

Set this equal to $0$, "distribute" the sum, and we can see

$$\frac{1}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i)] = 0 \implies w_0^* = \bar{y} - w_1^* \bar{x}$$

where $\bar{x}, \bar{y}$ represent the means of $x, y$.

### Minimizing MSE - simple linear regression (4)

Now, the slope:

$$ MSE(w_0, w_1) = \frac{1}{n} \sum_{i=1}^n [y_i - (w_0 + w_1 x_i) ] ^2 $$


$$ \frac{\partial MSE}{\partial w_1} = \frac{1}{n}\sum_{i=1}^n  2(y_i - w_0 -w_1 x_i)(-x_i)$$

$$  \implies -\frac{2}{n} \sum_{i=1}^n x_i (y_i - w_0 -w_1 x_i)  = 0$$

::: notes


### Minimizing MSE - simple linear regression (5)

This is equivalent to:

$$  \sum_{i=1}^n x_i e_i  = 0$$


### Minimizing MSE - simple linear regression (6)

Two conditions, 


$$  \sum_{i=1}^n e_i  = 0,  \sum_{i=1}^n x_i e_i  = 0$$

where 

$$ e = y_i - w_0 - w_1 x_i $$

### Minimizing MSE - simple linear regression (7)

Which we expand into

$$  \sum_{i=1}^n  y_i = n w_0 + \sum_{i=1}^n x_i w_1 $$

$$  \sum_{i=1}^n x_i y_i = \sum_{i=1}^n  x_i w_0 + \sum_{i=1}^n x_i^2 w_1 $$

### Minimizing MSE - simple linear regression (8)

Divide

$$  \sum_{i=1}^n  y_i = n w_0 + \sum_{i=1}^n x_i w_1 $$ 

by $n$, we find the intercept

$$w_0 = \frac{1}{n} \sum_{i=1}^n y_i - w_1 \frac{1}{n} \sum_{i=1}^n x_i $$

### Minimizing MSE - simple linear regression (9)

$$w_0 = \frac{1}{n} \sum_{i=1}^n y_i - w_1 \frac{1}{n} \sum_{i=1}^n x_i $$

$$ w_0 = \bar{y} - w_1 \bar{x} $$

where sample mean $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

### Minimizing MSE - simple linear regression (10)

To solve for $w_1$:  Multiply 

$$  \sum_{i=1}^n  y_i = n w_0 + \sum_{i=1}^n x_i w_1 $$


by $\sum x_i$, and multiply

$$  \sum_{i=1}^n x_i y_i = \sum_{i=1}^n  x_i w_0 + \sum_{i=1}^n x_i^2 w_1 $$


by $n$.

### Minimizing MSE - simple linear regression (11)


$$  \sum_{i=1}^n x_i \sum_{i=1}^n  y_i = n \sum_{i=1}^n x_i w_0 + (\sum_{i=1}^n x_i)^2 w_1 $$


$$  n \sum_{i=1}^n x_i y_i = n \sum_{i=1}^n  x_i w_0 + n \sum_{i=1}^n x_i^2 w_1 $$

Subtract the first equation from the second to get...

### Minimizing MSE - simple linear regression (12)

$$  n \sum_{i=1}^n x_i y_i - \sum_{i=1}^n x_i \sum_{i=1}^n  y_i = n \sum_{i=1}^n x_i^2 w_1  - (\sum_{i=1}^n x_i)^2 w_1 $$

$$ = w_1 \left( n \sum_{i=1}^n x_i^2 - (\sum_{i=1}^n x_i)^2 \right) $$

:::

### Minimizing MSE - simple linear regression (13)

Solve for $w_1^*$:

$$ w_1^*  = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) }{\sum_{i=1}^n (x_i - \bar{x})^2}$$

### Minimizing MSE - simple linear regression (14)

The slope coefficient is the ratio of *sample covariance* $\sigma_{xy}$ to *sample variance* $\sigma_x^2$:

$$ \frac{\sigma_{xy}}{\sigma_x^2} $$

where

$$ \sigma_{xy} = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}), \quad
\sigma_x^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x}) ^2$$

### Minimizing MSE - simple linear regression (15)

We can also express it as

$$ \frac{r_{xy} \sigma_y}{\sigma_x} $$

where sample correlation coefficient 
$r_{xy} = \frac{\sigma_{xy}}{\sigma_x \sigma_y}$.

::: notes

(Note: from Cauchy-Schwartz law, $|s_{xy}| < s_x s_y$, we know $r_{xy} \in [-1, 1]$)


:::

::: {.cell .markdown}

### MSE for optimal simple linear regression

$$MSE(w_0^*, w_1^1) = \sigma_y^2 - \frac{\sigma_{xy}^2}{\sigma_{x}^2}$$



and

$$\frac{MSE(w_0^*, w_1^1)}{\sigma_y^2} =  1- \frac{\sigma_{xy}^2}{\sigma_{x}^2 \sigma_{y}^2}$$


where the ratio on the right is the *coefficient of determination*, R2.

::: notes

* the ratio on the left is the *fraction of unexplained variance*: of all the variance in $y$, how much is still "left" after our model accounts for some of it? (best case: 0)
* R2 close to 1 is a very good model.

:::


:::



::: notes



### Linear regression - what can go wrong?

* Relationship may not actually be linear (may be addressed by non-linear transformation - future lecture)
* Violation of additive assumption (need interaction terms)
* "Tracking" in residuals (e.g. time series)
* Outliers - may be difficult to spot - may have outsize effect on regression line and/or $R^2$
* Collinearity  

:::


::: notes


## Maximum likelihood

In a linear model, if the errors belong to a normal distribution the least squares estimators are also the maximum likelihood estimators.


:::


