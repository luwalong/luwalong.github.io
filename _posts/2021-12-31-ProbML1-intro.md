---
title: Probabilistics Glossary for Machine Learning
tags: ProbabilisticMachineLearning Probabilitics Fundamentals
mathjax_autoNumber: true
---

Here we will skim the fundamental concepts from probabilistics to understand
machine learning in probabilistic scope. This document is heavily based on Kevin P. 
Murphy's *"Probabilistic Machine Learning: An Introduction*[^1] chapter 2, yet 
the core idea I got is from [Prof. Dr. Joachim M. Buhmann](https://inf.ethz.ch/people/person-detail.buhmann.html)'s 
*Advanced Machine Learning* and *Statistical Learning Theory*.
Herzlich danke für dear scholars in advance to proceed the contents.

- - -

## Bayes' Rule 

The axiom of **Bayesian inference**. Here we go for the basic terms of the elements.

$$ p(H=h | Y=y) = \frac{p(H=h) p(Y=y | H=h)}{p(Y=y)} $$

Consider $H$ as the *hidden* random variable (often, *model parameters* in ML) 
and $Y$ as the *observed* data (it's just a convention not to use $O$ but $Y$ 
when we call the ground truth labels or any of immutable properties). Bayes' 
rule is to calculate the probability distribution of the unknown quantity $H$ 
given some observed data $Y$. Few terms to keep in mind here:


Prior distribution $p(H)$
: can be considered as a *regularisation* term, prior distribution is what we know
about possible values of $H$ before we observe any data.

Observation distribution $p(Y | H=h)$
: a distribution of $Y$ under the assumption of $H=h$. This distribution is
directily linked to...

Likelihood $p(Y=y | H=h)$
: a probability (*not a distribution*) of observing $y$ under the $H=h$ condition.
Note that this is a function of $h$ but is not a probaility distribution since
the sum does not go 1.

Evidence or Marginal likelihood $p(Y=y)$
: marginalised likelihood, literally. This constant is usually treated as just a
normalisation denominator; often unknown.

Posterior distribution $p(H=h | Y=y)$
: think of that the data is existed before the inference, so calculating the 
probability of $H=h$ is *post* to the observed data. This represents the *belief
state* about $H$.


Using Bayes' rule, we perform **Bayesian inference** to estimate the unknown quantity
given the observed data. We can estimate the best $h^*$, based on argmax posterior,
is called **Maximum a posterior (MAP)** estimator. 

On the other hand, estimator basd on the likelihood is called **Maximum likelihood (ML)**
estimator. At this point of view called *frequentism*, there is no regularisation term 
and the estimator becomes consistent, asymptotically normal, and asympotically efficient.
We will cover this later on.


## Binary Classification

In the binary classification, we assume both the latent truth and observed data
have dichotomic states, saying, 0 or 1. Denote $H$ be the truth and $Y$ be the
observed result. Check this **confusion matrix**:

|---+---+---|
| | $Y=0$ | $Y=1$ |
|---|:---:|:---:|
| $H=0$ | True Negative Rate (TNR), *Specificity* | False Positive Rate (FPR), *Type I Error* |
|===+===+===|
| $H=1$ | False Negative Rate (FNR), *Type II Error* | True Positive Rate (TPR), *Sensitivity* |
|---|---|---|

All those rates are calculated based on the likelihood, $p(Y|H)$, a.k.a., in row-wise sense.
The terms and important metrics to remember are:

Prevalence $p(H=1)$
: as we set the prevalence, we know the prior of the hidden random variable $H$
as we only have $H=0$ and $H=1$ states.

Precision $p(H=1 | Y=1)$
: a posterior of $H=1$ given the observed data is positive. How many true positives
in the positively-observed data.

Recall $p(Y=1 | H=1) =$ TPR
: a likelihood of $Y=1$ when the latent truth is positive. How much of positive cases
is covered by the data.

Accuracy $p(Y=H)$
: a probability where the states are identical. Not conditioned on any.


### Logistic Regression

One of the most classic model for binary classification is *logistic regression* with
the following form:

$$ p(y | \mathbf{x}; \mathbf{\theta}) = \text{Ber} \left( y | \sigma(\mathbf{w}^\intercal \mathbf{x} + b) \right)$$

where

$$ \sigma(a) = \frac{1}{1 + e^{-a}} $$

is called **sigmoid** or **logistic function**, and $\text{Ber}(y|\mu)$ is the
Bernulli distribution with mean $\mu$. Not only in binary classifications,
we set the **decision boundary** to decide whether the prediction is which class.
In this case, we set it to be the value $x^* $ satisfying $p(y=1 | x^* ; \mathbf{\hat{\theta}}) = 0.5$.


## Useful Distributions




[^1]: Murphy, Kevin P. "Probabilistic Machine Learning: An Introduction." available [online](https://probml.github.io/pml-book/book1.html)

