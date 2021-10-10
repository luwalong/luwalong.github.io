---
title: Interpretable Machine Learning - Pilot
tags: interpretable_machine_learning book_reading
---

To my beloved Swimmer pals and strangers: I want to introduce this wonderful
resource named [**Interpretable Machine Learning**](https://christophm.github.io/interpretable-ml-book/)
by Christoph Molnar. This book has comprehensive concepts about interpretability
on machine learning and deep learning, so I think it's worth to study together
to understand what the field is and what are happening nowadays.

I will generally follow the chapters of the book from top to bottom, but with
some additional resources which I think could help the understanding of some
concepts. In this pilot page, I'll go through the motivation and introduction
of this topic with the material of a NeurIPS 2020 tutorial, *“Explaining Machine 
Learning Predictions: State-of-the-art, Challenges, and Opportunities”* by H.
Lakkaraju et al. You can find the original source from [here](https://explainml-tutorial.github.io/neurips20). 


## Why do we need interpretable models?

So why? No one can deny that machine learning is literally everywhere in this
world. Especially from mid-2010 when deep learning models first emerged, almost
all challenging tasks' SoA solutions have been eplaced by those ominous looking stacked
layers of matrices and functions. Those are deep; deeeep like Mariana Trench.
The only difference between the trench and deep learning models is that the former
has fixed depth but the latter is constantly getting deeper and even bigger.

One similarity I want to stress here is that both are not revealing much of their
essence. We know too little about deep learning. We know that it can solve numerous
tasks, but have no idea for *why it does perform well*. Unlike 'traditional' 
statistical models, deep neural networks' interior is opaque -- cannot find the
meaning at the first sight.

This critical ignorance for the essence of the greatest power the world has is not
welcomed by DARPA[^1]. They concerned that when the models' domain is so delicate,
emboding the sensitive virtues, or related to human life, the slightest error can
bring disasterous consequence. Especially when a human take charge in the
model, they have zero-ish idea why the model performed how.
They denoted this problem as **XAI: eXplainable Artificial Intelligence** and 
declared it as one of the core programs of their branches:
> XAI is one of a handful of current DARPA programs expected to enable “third-wave AI \
systems”, where machines understand the context and environment in which they operate,  ...

From then, the various works related to explainability and interpretability
converged into the newly named field, XAI. We want the full control on what we
make, and benefit us with the reliable results.


## Case studies of ML failures

Cases are excerpted from Last Week in AI #104[^2].

### Bayerischer Rundfunk's study

You can see the whole media [here](https://web.br.de/interaktiv/ki-bewerbung/en/).
Although AI is believed to introduce less 'prejudice' than people do, the German
broadcasting company BR's study found that in some AI assistants used for assessing
the job seekers' video are affected by irrelevant features, such as wearing glasses
or headscarf or putting a painting on the wall behind, during the applicants' 
personality assessment. Surely, that the target model seems not trained 'fairly',
but should we note that there is no way to check this fairness unless we do post hoc
examination so far, and moreover, no policy is made for making fair models. 


### Can Computer Algorithms Learn to Fight Wars Ethically?

Not exactly a failure already happened, but this [WP article](https://www.washingtonpost.com/magazine/2021/02/17/pentagon-funds-killer-robots-but-ethics-are-under-debate/)
triggers the debate on AIs' decision making process. How would the robot handle
kids in the war situation? Should they perform inaction to drive people to action
in some cases? AI ethics and XAI are tightly connected in these themes, and to get
any plausible solution of this, certain amount of translucent decision process
is required for the AI.


## How do we achieve model understanding?

### 1. Build inherently interpretable predictable models

We can use decision tree or linear regression to boost the explainability
of the decision. However, those traditional approaches lack performance.

### 2. Explain pre-built models in a post-hoc manner

Since we usually encounter (deep) neural networks to examine, we need an *explainer*
to interpret the result of the model. The approaches can be either local or
global, regarding the case's sensitivity.


## Wrap up

Hope this short post motivated you to start a journey to interpretability of
machine learning. This is a fairly new field yet has utmost importance to our
future scenary:)


[^1]: Explainable Artificial Intelligence (XAI) (2018), M. Turek, [https://www.darpa.mil/program/explainable-artificial-intelligence](https://www.darpa.mil/program/explainable-artificial-intelligence)
[^2]: Last Week in AI #104 (2021), [https://lastweekin.ai/p/104](https://lastweekin.ai/p/104)
