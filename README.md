# NCTU\_106-2\_machine-learning
Assignments for machine-learning in NCTU

## Lab1 - rLSE & Newton's method
- Usage: cd lab1 && python3 main.py &lt;data file&gt; &lt;bases number&gt; &lt;lambda&gt;

## Lab2 - Naive Bayes & Beta-Binomial Online Learning
### Naive Bayes
- Usage: cd lab2 && python3 main.py \[option \[training images \[training labels \[testing images \[testing labels\]\]\]\]\]
  - option 0: Discrete - 256 in 32 bins
  - option 1: Continuous - Standard Normal Distribution

### Binomial
- Usage: cd lab2 && python3 binomial.py \[data \[a \[b\]\]\]
- Online learning with Beta Distribution as prior

## Lab3 - Data Generator & Online Learning
### Data Generator
- Usage: cd lab3 && python3 generator.py &lt;mode&gt; &lt;mean var | a w&gt;
  - mode 0: Gaussian Distribution data generator y~N(mean, var)
  - mode 1: Polynomial Basis Linear Model data generator y = Phi(x)\*w + e, e~N(0, a), -10&lt;x&lt;10
  - mode 2: Plot for mode 0 with 10000 data
  - mode 3: Plot for mode 1 with 10000 data

### Gaussian Estimator
- Usage: cd lab3 && python3 estimator.py &lt;number of data&gt; &lt;mean&gt; &lt;var&gt;

### Bayesian Linear Regression
- Usage: cd lab3 && python bayesian.py &lt;precision&gt; &lt;a&gt; &lt;w&gt;
  - a and w is for polynomial basis linear model data generator
- Assume we have known the data variance

## Lab4 - Logistic Classifier & EM clustering
### Logistic Classifier
- Usage: cd lab4 && python3 logistic.py &lt;n&gt; &lt;mx1&gt; &lt;vx1&gt; &lt;my1&gt; &lt;vy1&gt; &lt;mx2&gt; &lt;vx2&gt; &lt;my2&gt; &lt;vy2&gt;
- Distinguish two normal-distribution data set

### EM clustering
- Usage: cd lab4 && python3 em.py
- Cluster MNIST
