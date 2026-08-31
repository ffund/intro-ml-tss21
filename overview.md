

## Course overview

<table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 100%;">
  <thead style="background-color: #f2f2f2;">
    <tr>
      <th>Topic</th>
      <th>Math prerequisites</th>
      <th>Textbook references</th>
      <th>StatQuest videos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Linear regression</b></td>
      <td>
        Matrix notation, matrix–vector multiplication (IALA Sec. II.5)<br>
        Inner product/dot product (IALA Sec. I.1)<br>
        Norm of a vector (IALA Sec. I.3)<br>
        Matrix inverse (IALA Sec. II.11)<br>
        Derivatives and optimization (IALA App. C)
      </td>
      <td>
        ISL: Sec. 3.1–3.3; Sec. 7.3 (basis functions)<br>
        ESL: Sec. 3.2 (linear regression and least squares); Sec. 5.1 (basis expansions)<br>
        PRML: Sec. 3.1, especially Sec. 3.1.1–3.1.2<br>
        IALA: Sec. 12.2 (OLS derivation)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=nk2CQITm_eo">Linear regression, clearly explained!!!</a><br>
        <a href="https://www.youtube.com/watch?v=zITIFTsivN8">Multiple regression, clearly explained!!!</a><br>
        <a href="https://www.youtube.com/watch?v=wl1myxrtQHQ">The Chain Rule</a>
      </td>
    </tr>
    <tr>
      <td><b>Gradient descent</b></td>
      <td>
        Derivatives and optimization (IALA App. C)<br>
        Complexity of algorithms, especially vector/matrix operations (IALA App. B, I.1, II.6)
      </td>
      <td>
        ISL: Sec. 10.7.2 (SGD in a neural-network context)<br>
        ESL: Sec. 10.10.1 (steepest descent in a gradient-boosting context)<br>
        PRML: Sec. 3.1.3 (SGD and sequential learning for linear regression); Sec. 5.2.4 (general gradient-descent optimization)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=sDv4f4s2SB8">Gradient descent, step-by-step</a>
      </td>
    </tr>
    <tr>
      <td><b>Error decomposition and bias–variance tradeoff</b></td>
      <td>
        Expectation of a random variable (linearity, constants)<br>
        Variance of a random variable<br>
        Independence of random variables
      </td>
      <td>
        ISL: Sec. 2.2.2<br>
        ESL: Sec. 2.9; Sec. 7.2–7.3<br>
        PRML: Sec. 3.2
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=EuBBz3bI-aA">Machine Learning Fundamentals: Bias and Variance</a>
        </td>
    </tr>
    <tr>
      <td><b>Model selection</b></td>
      <td>—</td>
      <td>
        ISL: Sec. 5.1.1–5.1.4 (cross-validation); Sec. 7.1, 7.3–7.4 (polynomial, basis, and spline model complexity)<br>
        ESL: Sec. 5.2 (piecewise polynomials and splines); Sec. 7.2–7.3, 7.10<br>
        PRML: Sec. 1.3
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=fSytzGwwBVw">Cross validation</a>
        </td>
    </tr>
    <tr>
      <td><b>Regularization</b></td>
      <td>
        Derivatives and optimization (IALA App. C)<br>
        Norm of a vector (IALA Sec. I.3)
      </td>
      <td>
        ISL: Sec. 6.2.1–6.2.3 (ridge, lasso, and tuning parameters)<br>
        ESL: Sec. 3.4.1–3.4.3 (ridge, lasso, and comparison)<br>
        PRML: Sec. 3.1.4 (L2/weight-decay regularization; not lasso)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=Q81RR3yKn30">Ridge regression (L2 regularization)</a><br>
        <a href="https://www.youtube.com/watch?v=NGf0voTMlcs">Lasso regression (L1 regularization)</a><br>
        <a href="https://www.youtube.com/watch?v=1dKRdX9bfIo">Elasticnet regression (L1 and L2)</a>
        </td>
    </tr>
    <tr>
      <td><b>Logistic regression</b></td>
      <td>
        Probability mass functions<br>
        Bernoulli random variable<br>
        Independence of samples<br>
        Conditional probability<br>
        Joint probability
      </td>
      <td>
        ISL: Sec. 4.3.1–4.3.5; Sec. 4.4.4 (Naive Bayes)<br>
        ESL: Sec. 4.4–4.4.1 (logistic regression); Sec. 6.6.3 (Naive Bayes)<br>
        PRML: Sec. 4.3.2–4.3.4; Sec. 4.2.3 (discrete generative classification and Naive-Bayes-style modeling)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=yIYKR4sgzI8">Logistic Regression</a><br>
        <a href="https://www.youtube.com/watch?v=vN5cNN2-HWE">Logistic Regression Details Pt1: Coefficients</a><br>
        <a href="https://www.youtube.com/watch?v=BfKanl1aSG0">Logistic Regression Details Pt 2: Maximum Likelihood</a><br>
        <a href="https://www.youtube.com/watch?v=ARfXDSkQf1Y">Odds and Log(Odds), Clearly Explained!!!</a><br>
        <a href="https://www.youtube.com/watch?v=4jRBRDbJemM">ROC and AUC, Clearly Explained!</a><br>
        <a href="https://www.youtube.com/watch?v=Kdsp6soqA7o">Confusion matrix</a><br>
        <a href="https://www.youtube.com/watch?v=vP06aMoz4v8">Sensitivity and Specificity</a>
        </td>
    </tr>
    <tr>
      <td><b>K-nearest neighbor</b></td>
      <td>
        Expectation of a random variable (linearity, constants)<br>
        Variance of a random variable<br>
        Independence of random variables
      </td>
      <td>
        ISL: Sec. 2.2.3, 3.5<br>
        ESL: Sec. 2.3.2; Sec. 13.3–13.5<br>
        PRML: Sec. 2.5.2 (nearest-neighbor methods; less direct coverage of KNN regression)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=HVXime0nQeI">K-nearest neighbors, Clearly Explained</a>
      </td>
    </tr>
    <tr>
      <td><b>Decision trees and ensembles</b></td>
      <td>
        Variance of a random variable<br>
        Independence of random variables<br>
        Variance of sum of random variables
      </td>
      <td>
        ISL: Sec. 8.1.1–8.1.4, 8.2.1–8.2.3<br>
        ESL: Sec. 8.7 (bagging); Sec. 9.2 (trees); Sec. 10.1–10.12 (boosting); Sec. 15.1–15.4 (random forests)<br>
        PRML: Sec. 14.2–14.4 (committees, boosting, and tree-based models)
      </td>
      <td>
      <a href="https://www.youtube.com/watch?v=_L39rN6gz7Y">Decision Trees, Clearly Explained!!!</a><br>
      <a href="https://www.youtube.com/watch?v=wpNl-JwwplA">Decision Trees, Part 2 - Feature Selection and Missing Data</a><br>
      <a href="https://www.youtube.com/watch?v=g9c66TUylZ4">Regression Trees, Clearly Explained!!!</a><br>
      <a href="https://www.youtube.com/watch?v=D0efHEJsfHo">How to Prune Regression Trees, Clearly Explained!!!</a><br>
      <a href="https://www.youtube.com/watch?v=J4Wdy0Wc_xQ">Random Forests Part 1:  Building, Using and Evaluating</a><br>
      <a href="https://www.youtube.com/watch?v=LsK-xG1cLYA">AdaBoost, Clearly Explained</a>
      </td>
    </tr>
    <tr>
      <td><b>Support Vector Machines, Kernels, Other Kernel-Based Models</b></td>
      <td>
        Expectation of a random variable (linearity, constants)<br>
        Variance of a random variable<br>
        Independence of random variables<br>
        Conditional distributions<br>
        Gaussian distribution
      </td>
      <td>
        ISL: Sec. 9.1–9.3<br>
        ESL: Sec. 12.2–12.3<br>
        PRML: Sec. 6.2 (kernels); Sec. 7.1–7.1.3 (maximum-margin/SVM classification); Sec. 6.4.2–6.4.3 (Gaussian-process regression and hyperparameter learning)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=efR1C6CvhmE">Support Vector Machines Part 1 (of 3): Main Ideas!!!</a><br>
        <a href="https://www.youtube.com/watch?v=Toet3EiSFcM">SVM with Polynomial kernel</a><br>
        <a href="https://www.youtube.com/watch?v=Qc5IyLW_hns">SVM with RBF kernel</a>
      </td>
    </tr>
    <tr>
      <td><b>Neural networks</b></td>
      <td>
        Derivatives and optimization (IALA App. C)<br>
        Chain rule for multivariable functions
      </td>
      <td>
        ISL: Sec. 10.1–10.2, 10.7.1<br>
        ESL: Sec. 11.3–11.5<br>
        PRML: Sec. 5.1–5.3
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=CqOfi41LfDw">Neural Networks Pt. 1: Inside the Black Box</a><br>
        <a href="https://www.youtube.com/watch?v=IN2XmBhILt4">Neural Networks Pt. 2: Backpropagation Main Ideas</a><br>
        <a href="https://www.youtube.com/watch?v=iyn2zdALii8">Backpropagation Details Part 1</a><br>
        <a href="https://www.youtube.com/watch?v=GKZoOHXGcLo">Backpropagation Details Part 2</a><br>
        <a href="https://www.youtube.com/watch?v=68BZ5f7P94E">Neural Networks Pt. 3: ReLU In Action!!!</a><br>
        <a href="https://www.youtube.com/watch?v=83LYR-1IcjA">Neural Networks Pt. 4: Multiple Inputs and Outputs</a><br>
        <a href="https://www.youtube.com/watch?v=KpKog-L9veg">Neural Networks Part 5: ArgMax and SoftMax</a><br>
        <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Neural Networks Part 6: Cross Entropy</a><br>
        <a href="https://www.youtube.com/watch?v=xBEh66V9gZo">Neural Networks Part 7: Cross Entropy Derivatives and Backpropagation</a><br>
        <a href="https://www.youtube.com/watch?v=FHdlXe1bSe4&t=943s">Introduction to PyTorch</a>
      </td>
    </tr>
    <tr>
      <td><b>Deep neural networks, convolutional neural networks</b></td>
      <td>—</td>
      <td>
        ISL: Sec. 10.2, 10.3.1–10.3.5, 10.7.2–10.7.4, 10.8<br>
        ESL: Sec. 11.3–11.5 (classical neural-network architecture and training only; not modern CNNs, augmentation, transfer learning, or double descent)<br>
        PRML: Sec. 5.5.6 (convolutional networks; older treatment, not modern deep-learning practice)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=HGwBXDKFk9I&t=1s">Image Classification with Convolutional Neural Networks (CNNs)</a>
      </td>
    </tr>
    <tr>
      <td><b>Unsupervised learning</b></td>
      <td>
        Expectation and variance of random variables<br>
        Covariance and covariance matrices<br>
        Eigenvalues and eigenvectors<br>
        Probability distributions<br>
        Joint and conditional probability
      </td>
      <td>
        ISL: Sec. 12.2.1–12.2.5 (PCA); Sec. 12.4.1–12.4.3 (K-means and hierarchical clustering; partial coverage of the lecture)<br>
        ESL: Sec. 14.3.1–14.3.12 (clustering); Sec. 14.5.1–14.5.5 (PCA and related dimensionality reduction); Sec. 14.8–14.9 (further embeddings and dimensionality reduction)<br>
        PRML: Sec. 2.5.1, 9.2 (density estimation and Gaussian mixtures); Sec. 9.1 (K-means); Sec. 12.1.1–12.1.4 (PCA)
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=HMOI_lkzW08">PCA main ideas in only 5 minutes!!!</a><br>
        <a href="https://www.youtube.com/watch?v=FgakZw6K1QQ">Principal Component Analysis (PCA), Step-by-Step</a><br>
        <a href="https://www.youtube.com/watch?v=4b5d3muPQmA">K-means clustering</a><br>
        <a href="https://www.youtube.com/watch?v=viZrOnJclY0">Word Embedding and Word2Vec, Clearly Explained!!!</a>
      </td>
    </tr>
    <tr>
      <td><b>Reinforcement learning</b></td>
      <td>
        Derivatives and optimization (IALA App. C)<br>
        Derivatives of exponential and log functions<br>
        Expectation of random variables<br>
        Conditional probability
      </td>
      <td>
      RL: Sec. 2.1, 2.2, 2.4, 2.5, 2.8, 6.1, 6.5, 13.3
      </td>
      <td>
        <a href="https://www.youtube.com/watch?v=Z-T0iJEXiwM">Reinforcement Learning: Essential Concepts</a><br>
      </td>
    </tr>
    <tr>
      <td><b>Recommender systems</b></td>
      <td>
        Matrix notation and matrix multiplication<br>
        Dot products and cosine similarity<br>
        Derivatives and optimization<br>
        Expectation of random variables
      </td>
      <td>MMDS: Ch. 9, Recommendation Systems</td>
      <td>—</td>
    </tr>
  </tbody>
</table>

<br>

<p><b>Legend:</b></p>
<ul>
  <li><b><a href="https://drive.google.com/file/d/1ajFkHO6zjrdGNqhqW1jKBZdiNGh_8YQ1/view">ISL</a></b> – <i>Introduction to Statistical Learning</i> (James et al.)</li>
  <li><b><a href="https://hastie.su.domains/ElemStatLearn/">ESL</a></b> – <i>Elements of Statistical Learning</i> (Hastie, Tibshirani, Friedman)</li>
  <li><b><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf">PRML</a></b> – <i>Pattern Recognition and Machine Learning</i> (Bishop)</li>
  <li><b><a href="https://web.stanford.edu/~boyd/vmls/vmls.pdf">IALA</a></b> – <i>Introduction to Applied Linear Algebra</i> (Boyd &amp; Vandenberghe)</li>
  <li><b><a href="http://www.mmds.org/#ver21">MMDS</a></b> – <i>Mining of Massive Datasets</i> (Leskovec, Rajaraman, Ullman)</li>
  <li><b><a href="http://incompleteideas.net/book/the-book-2nd.html">RL</a></b> – <i>Reinforcement Learning: An Introduction</i> (Sutton &amp; Barto)</li>
</ul>
