---
title: Interpretable Machine Learning - Neural Network Interpretation - (2) Detecting Concepts
tags: interpretable_machine_learning book_reading detecting_concepts TCAV
---

So from the previous post, we took a brief look over the pixel attribution
approaches. This time, we'll go through another linage of interpretability -
**concept-based approaches**. Let's start from what 'concept' is, and why we
need it. Again, this post follows the contents from [here](https://christophm.github.io/interpretable-ml-book/detecting-concepts.html).

## TCAV: Testing with Concept Activation Vectors

### Pitfalls of feature-based approaches

Pixel attribution or saliency map approaches are not as perfect as we initially
expected, but they seem okay - visualisable and explanable intuitively. Let's
see big two problems of them:

1. Feature values often are not user-friendly, or human-centric. The author of
   the original page makes an example that a single pixel from an imaged do not play a
   significant role for a person. That's right; we don't care much about those
   subtle pixel values. Moreover, we might not make an agreement for the
   intrepretation of certain degree of variance.
1. The expressiveness of a feature-based approach is limited by the number of
   features. That's obvious, and what do you want more? Consider a 2D cartesian
   coordinate system as an input set. What if the labels correlate with the
   quadrant where a datapoint lies?

**Concept** arose as an alternative to those feature-centric perspectives.
Concept can be either abstract or concrete, and has so high flexibility of
definition to make arbitrary groups. For example, color tone of images, theme,
or even your favour. So how do we embrace those subtle definitions?


### Concept Activation Vector (CAV)

Kim et al.[^1] suggested Testing with Concept Activation Vectors (TCAV) method.
The main concept of the method, Concept Activation Vector (CAV), denoted as
$v_l^C$, is defined from the concept $C$ and a neural network layer $l$. CAV is
basically a numerical representation that generalizes a concept in the activation
space.

We need to prepare two datasets: one is from the concept $C$ and another is from
the random set as a reference. Also, we should determine which layer $l$ to be a
target layer, and train a binary classifier (can use SVM or logistic regression)
upon it to separate the concept and random datasets. From the classifier, we get
the coefficient vector and denote it as a CAV $v_l^C$. *"Conceptual
sensitivity"* of given input $x$ is defined as a directional derivative of the
prediction in the direction of CAV:

$$ S_{C,k,l}(x) = \lim_{\epsilon \rightarrow 0} \frac{h_{l,k}(\hat{f}_l(x) + \epsilon v_l^C) - h_{l,k}(\hat{f}_l(x))}{\epsilon} = \nabla h_{l,k} (\hat{f}_l (x)) \cdot v_l^C $$

where $$\hat{f}_l$$ maps the input to the activation vector of the layer $l$ and
$h_{k,l}$ maps the activation vector to the logit output of class $k$.
Intuitively, the more similar the directions of gradient and CAV, the higher the
sensitivity they have.


### Testing with CAV

So we want to calculate the global level of interpretation. To obtain such
quantity, TCAV calculate the ratio of inputs with positive conceptual
sensitivities to the number of inputs for a class:

$$ TCAV_{C,k,l} = \frac{|x \in X_k : S_{C,k,l}(x) > 0|}{|X_k|} $$

Pretty simple, isn't it? The author brings an example of zebra image
classification. If the concept 'striped' shows $TCAV=0.8$ or the class 'zebra',
then it can be interpretable as 80% of predictions for 'zebra' are positively
influenced by the concept 'striped'!

As simple as it is, we need to do more stuffs to make this statistics more
reliable. To guard against spurious results from testing a class against
a particular CAV, Kim et al. propose the following simple statistical
significance test. Instead of training a CAV once, against a
single batch of random examples $N$, we can perform multiple
training runs, typically 500. A meaningful concept should
lead to $TCAV$ scores that behave consistently across training
runs.

Concretely we perform a two-sided $t$-test of the $TCAV$
scores based on these multiple samples. If we can reject the
null hypothesis of a $TCAV$ score of 0.5, we can consider
the resulting concept as related to the class prediction in a
significant way. Note that we can also perform a Bonferroni
correction for our hypotheses (at $p \lt \alpha/m$ with $m = 2$) to
control the false discovery rate further.

The authors also suggests Relative $TCAV$, which replaces the reference random
dataset to different concept group, $D$. In this way, we can gauge the relative
importance of multiple concepts.


## Pros and Cons

Well the concept of TCAV is fairly simple yet quantisable. That brings the first
pros: users don't need ML expertise to get the values. Also, TCAV provides
extremely high customisability in defining concepts from the whole data, and can
be applied to global explanation level. TCAV can be used in spotting a weak spot
of training model - e.g., if there is a classifier that predicts “zebra” with a high accuracy, TCAV, however, shows that the classifier is more sensitive towards the concept of “dotted” instead of “striped”. This might indicate that the classifier is accidentally trained by an unbalanced dataset, allowing you to improve the model by either adding more “striped zebra” images or less “dotted zebra” images to the training dataset.

Also there are some cons for the method. TCAV is known to perform worse on
shallower networks, since concepts are more separable in deeper layers[^2].
Plus, certain level of flexibility in concept definition is needed, and we're
not sure of the comparable performance on non-CV domain data.


[^1]: Kim, Been, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, and Fernanda Viegas. “Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav).” In International conference on machine learning, pp. 2668-2677. PMLR (2018).
[^2]: Alain, Guillaume, and Yoshua Bengio. “Understanding intermediate layers using linear classifier probes.” arXiv preprint arXiv:1610.01644 (2016).
