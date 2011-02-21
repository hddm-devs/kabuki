************
Introduction
************

:Date: February 20, 2011
:Author: Thomas Wiecki
:Contact: thomas_wiecki@brown.edu
:Web site: http://code.google.com/p/kabuki
:Copyright: This document has been placed in the public domain.
:License: Kabuki is released under the GPLv3.
:Version: 0.1

Purpose
=======

Kabuki is a Python toolbox that allows aids creation of hierarchical Bayesian models for cognitive science using PyMC. In many psychological and cognitive experiments multiple subjects are tested on the same task. Traditional model estimation methods (such as maximum likelihood) either fit a model to each individual subject (ignoring any similarities they will probably have) or to the group as a whole (ignoring individual differences). Due to common scarcety of data the first approach is not feasible in many circumstances. Ideally, our model captures that subjects are different but still similar. 

Hierarchical Bayesian modeling allows us to construct models that perfectly map the nature of the experiment -- subject parameter distributions are constrained by a group parameter distribution. Kabuki makes it extremely easy to apply this specific pattern to your data.

Features
========

* Simplicity: Simply decorate your class and let kabuki take care of constructing and estimating the model.  

* Flexibility in model specification: Kabuki tries not to constrain any models you'll want to create. You specify which parameters your model uses and what distribution the likelihoods have.

* Flexibility in model fitting: After specifiying your model you can tailor it to specific task conditions on the fly (e.g. parameter A in your model depends on the stimulus type that was used).

* Easy specification of models via configuration file fosters exchange of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and goodness-of-fit diagnostics.

* Kabuki comes pre-equipped with some example models that are widely usable such as a Bayesian ANOVA model for hypothesis testing.

Usage
=====

Getting started
===============
