---
title: Importance Sampling
tags: sampling_method importance_sampling variance_reduction
---

For some reason, I'm now working on the sampling methods to improve the models'
performance. We have tons of data, but using every row is pretty expensive and
shows slower convergence. After some research, I found an insightful reference
of training data sampling for neural networks. Although it's not clear if this
is applicable to my project, let's go through the brief introduction of the
method.

## Why importance sampling?

**Importance sampling** is based on the intuition that *not all samples are
equally important* to practitioners. Some outliers could just harm the model
performance, for example, and without knowing those, the model complexity could
dramatically increase to incorporate all samples. Hence, should some samples are
more important than others, we can focus on those stressed out ones first.

Importance sampling is applicable both to convex problems and neural networks,
and, of course, we're interested in the latter case more:) Generally, sampling
method helps the model training to reduce variance of the estimator.

## Importance sampling in deep learning

### Curriculum learning

Bengio et al.[^1] suggested *curriculum learning* for training deep neural
networks in the early era. This is somehow mimicking the human learning
process--by providing gradually harder problems, the model converges better and
faster. Well, we cannot say this is best way to train the model since there are
directly opposite ways reported as working well. In those cases, easy examples
are considered as *non-informative*.

It has been more than 10 years, and you know, 10 years in this field means a
lot--the examples paper presented are already somehow outdated, such as
skip-gram language model. Fortuantely, according to [^4], curriculum learning is
still an active field of this line.

### Loss-based sampling

Another line of work is *loss-based sampling*. [^2] and [^3] use the loss
history for the sampling distribution. The former is from Google, and provides
nice intuition of for choosing better samples (replays) by 'surpriseness' for reinforcement learning in game playing. However, both have limitations of expensive
hyperparameter tunings for handling "stale" importance scores.

### Importance score by gradient upper bound

EPFL folks[^5] presented interesting approach to the imporatance sampling field.
Following the notations from their paper (and probably you'll notice the
notations quickly), the goal of deep learning training is to find

$$ \theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}
(\Psi(x_i;\theta), y_i) $$

When we use SGD to train the model with learning rate $\eta$ based on the
sampling distribution $p_1^t, \cdots, p_N^t$, re-scaling coefficients $w_1^t,
\cdots, w_N^t$, and $I_t$ be the data point sampled at step $t$, we have $P(I_t
= i) = p_i^t$ and

$$ \theta_{t+1} = \theta_t - \eta w_{I_t} \nabla_{\theta_t}
\mathcal{L}(\Psi(x_{I_t}; \theta_t), y_{I_t}) $$

Here, plain SGD with uniform sampling is equivalent to using $w_i^t = 1$ and
$p_i^t = 1/N,\ \forall t,i$. Let's define convergence speed $S$:

$$ S = -\mathbb{E}_{P_t} \left[ \| \theta_{t+1} - \theta^* \|_2^2 - \| \theta_t - \theta^* \|_2^2 \right] $$

If we set $w_i = \frac{1}{N p_i}$ and $G_i = w_i \nabla_{\theta_t}  \mathcal{L}(\Psi(x_{I_t}; \theta_t), y_{I_t})$, then according to the authors,

$$ S = 2\eta (\theta_t - \theta^*) \mathbb{E}_{P_t} [G_{I_t}] - \eta^2
\mathbb{E}_{P_t} [G_{I_t}]^T \mathbb{E}_{P_t} [G_{I_t}] - \eta^2 tr(\mathbb{V}_{P_t} [G_{I_t}]) $$

The first two terms are the speed of batch gradient descent, but minimising the
last trace term exactly takes too expensive compuatations. The author suggests
to set a bound $$ \hat{G}_i \geq \| \nabla_{\theta_t} \mathcal{L} (\Psi (x_i;
\theta_t), y_i) \|_2 $$, and due to

$$ \arg\min_P tr(\mathbb{V}_{P_t} [G_{I_t}]) = \arg\min_P \mathbb{E}_{P_t} [\|
G_{I_t} \|_2^2] $$

the relaxed optimisation problem as:

$$ \min_P \mathbb{E}_{P_t} [\| G_{I_t} \|_2^2 ] \leq \min_P \mathbb{E}_{P_t} [w_{I_t}^2 \hat{G}_{I_t}^2]. $$

After the long calculation, the authors suggests the proxy of the upper bound
$\hat{G}_i$. See the paper for the full calculus, and bring the key idea from
here only:) By calculating $\hat{G}_i$, we can consider those as the *importance
score*, and weigh more to samples with higher scores.

Unlike previously suggested methods, this approach guarantees the theoretical
basis on the approximation of reduced variance from the sampling. Also, this
is beneficial by having much less computation cost and hyperparameter tuning.
However, just as other theoretically guaranteed approaches do, this results are
generally more conservative than the reality.



## Reference
[^1]: Bengio, Yoshua, et al. "Curriculum learning." Proceedings of the 26th annual international conference on machine learning. 2009.
[^2]: Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
[^3]: Loshchilov, I. and Hutter, F. "Online batch selection for faster training of neural networks." arXiv preprint arXiv:1511.06343 (2015).
[^4]: Portelas, Rémy, et al. "Automatic curriculum learning for deep rl: A short survey." arXiv preprint arXiv:2003.04664 (2020).
[^5]: Katharopoulos, Angelos, and François Fleuret. "Not all samples are created equal: Deep learning with importance sampling." International conference on machine learning. PMLR, 2018.
