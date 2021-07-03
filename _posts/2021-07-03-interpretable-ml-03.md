---
title: Interpretable Machine Learning - Interpretable Models - (3) RuleFit
tags: interpretable_machine_learning book_reading rulefit
---

RuleFit[^1] is the extension of [decision rules](https://christophm.github.io/interpretable-ml-book/rules.html), which can be easily summarised: classifying instances by several `IF-THEN` statements. We will skip the introduction for the decision rules, so please refer Molnar's page if you're interested in the model. Decision rules technique has their own distinct strategies to enhance interpretability and accuracy, so would worth take a look for the reference before we go into RuleFit.


## RuleFit

RuleFit is a quite recent approach to make the model as simple as linear regression, but able to capture the interactions between the variables at the same time. The core ideas of RuleFit are *to use the decision rules taken from the decision trees as features* and *to fit a sparse linear model* from them. Not tangible yet? Me neither. Let's drift into the details.


## Theory behind

The model consists of two parts: (1) the rules from decision trees, and (2) a linear model taking both inputs and the rules as new features. You see, "RuleFit" is named after so obvious reasons. 

### 1. Rule generation

*Rules* in RuleFit has the same `IF-THEN` format. Any route taken from a decision tree can be interpreted as a decision rule with conjuncted predicates. The original paper suggests to build several decision trees by gradient boosting and aggregate those by any tree ensemble algorithm, e.g., bagging, random forest, AdaBoost, etc. This can be formulated:

$$ f(X) = a_0 + \sum_{m=1}^{M} a_m f_m (X), $$

where $M$ is the number of trees, and $f_m(X)$ is the prediction of the $m$-th tree of the instance $X$. Each rule $r_m (X)$ takes the form of:

$$ r_m (X) = \prod_{j \in T_m} I_{\{ x_j \in s_{jm} \}}, $$

where $T_m$ is the subset of features used in $m$-th tree and $s_{jm}$ is the criterion of each feature used in the rule. After all, there will be $K$ rules built by the ensemble of $M$ trees with $t_m$ terminal nodes:

$$ K = \sum_{m=1}^M 2(t_m - 1). $$

Note that those rules can have arbitrary lengths (suggested by authors by training the random depth decision trees), and interpretable as the binary features from complex interactions between the features.

### 2. Sparse linear model

Since there will be so many rules obtained from the decision trees, we need to winsorise the original features so that they are more robust against outliers:

$$ l^*_j (x_j) = \min (\delta^+_j, \max(\delta^-_j, x_j)), $$

where $\delta^+$ and $\delta^-$ are $\delta$-quantiles of the variable. As a rule of thumb, we can set the $\delta=0.025$. After then, we need to normalise the features so that they have the same prior importance as a typical decision rule:

$$ l_j (x_j) = 0.4 \cdot \frac{l^*_j (x_j)}{std(l^*_j (x_j))} $$

Here, 0.4 is the average standard deviation of rules with a uniform support distribution of $s_k \sim U(0,1)$. Here, *support* $s_k$ is the portion of the data covered by the rule. Using all the winsorised features and rules, RuleFit trains the sparse linear model with the following format:

$$ \hat{f}(x) = \hat\beta_0 + \sum_{k=1}^K \hat\alpha_k r_k (x) + \sum_{j=1}^p \hat\beta_j l_j(x_j) $$

$$ (\{\hat\alpha\}_1^K, \{\hat\beta\}_1^p) = \arg\min \sum_{i=1}^n \mathcal{L}(y^{(i)}, f(x^{(i)})) + \lambda \left( \| \alpha \|_1 + \| \beta \|_1 \right) $$

### (optional) Feature importance

Recap the feature importance of the linear model is defined as the absolute weight divided by the standard deviation of the variable. The authors defined the importance of each rule as the following:

$$ I_i = \begin{cases}
|\hat{\beta}_i| \cdot std(l_i (x_i)) & \text{if } x_i \text{ is a standardised predictor} \\
|\hat\alpha_i| \cdot \sqrt{s_i (1-s_i)} & \text{if } x_i \text{ is a decision rule term}
\end{cases} $$

Note that a single feature can appear not only in the original set of features, but also in several other rules. The importance $J_j (x)$ of a feature can be measured for each individual prediction:

$$ J_j (x) = I_j(x) + \sum_{x_j \in r_k} I_k (x) / m_k $$

where $m_k$ is the number of features in the rule $r_k$. Summing up all the feature importances results in the global feature importance.


## Wrap up

RuleFit is rather a simple method combining the decision trees and linear regression. However, the power of the model is quite impressive: it is able to do both classificiation and regression, and handle the variables' interactions. Intrepretation of the model is straightforward, as its two ancestors do. 

The drawbacks, however, exist. There might be too many rules created and allocated with non-zero weights, which causes degrading interpretability as the number of features increases. Also, this model inherits not only the merits of linear models, but the bad things as well - weight interpretation is not always intuitive. Think of there are many overlapping rules but having similar importances. This model is claimed to perform as good as random forest does (according to authors), but has its own limitation.


[^1]: Friedman, Jerome H., and Bogdan E. Popescu. "Predictive learning via rule ensembles." The Annals of Applied Statistics 2.3 (2008): 916-954.
