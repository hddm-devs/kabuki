************
Introduction
************

:Date: March 8 19, 2012
:Author: Thomas Wiecki, Imri Sofer
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu
:Web site: http://github.com/hddm-devs/kabuki
:Copyright: This document has been placed in the public domain.
:License: Kabuki is released under the GPLv3.
:Version: 0.3dev

Purpose
=======

Kabuki is a Python toolbox that aids creation of hierarchical Bayesian models for cognitive science using PyMC. In many psychological and cognitive experiments multiple subjects are tested on the same task. Traditional non-hierarchical model estimation methods (such as maximum likelihood) either fit a model to each individual subject (ignoring any similarities they have) or to the group as a whole (ignoring individual differences). Due to common scarcety of data the first approach is not feasible in many circumstances. A better model would capture that subjects are different but still similar. 

Hierarchical Bayesian modeling allows us to construct models that map the nature of the experiment -- individual subject parameters are themselves distributed according to a group parameter distribution. Kabuki makes it extremely easy to create various models of this structure.

Features
========

* Simplicity: Define 3 functions that return the parameters of your model. The complete model creation is taken care of by kabuki.

* Flexibility in model specification: Kabuki tries not to constrain any models you'll want to create. You specify which parameters your model uses and what distribution the likelihoods have.

* Flexibility in model fitting: After specifiying your model you can tailor it to specific task conditions on the fly (e.g. parameter A in your model depends on the stimulus type that was used).

* Easy specification of models via configuration file fosters exchange of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and goodness-of-fit diagnostics.

* Kabuki comes pre-equipped with some example models that are widely usable such as a Bayesian ANOVA model for hypothesis testing.

The Problem
===========

Cognitive scientists use mathematical models to describe behavior. These models can be fit to behavioral data to make inference about latent variables. Normally, multiple subjects perform the same experiment and Maximum Likelihood (ML) is used to find the set of parameters that best describe the observed data of each individual subject. However, this method has several disadvantages (discussed in more detail below). One major drawback is that the estimation does not take into account that there are likely to be a similarities between subjects and what we learn about one we can apply to others. One alternative is to fit one model to all subjects however ignoring individual differences. Ideally, we want a method that allows us to portray the structure of our data as closely as possible onto the model.

The Solution
============

Hierarchical Bayesian modeling offers a remedy to this problem: Each subject parameter is constrained by a group parameter. Also, the full posterior distributions get estimated which allows to see how probable other parameter values are and allows for direct hypothesis testing that does not rely on t-tests or F-tests.
Kabuki

Kabuki is a Python library that builds on top of PyMC, a general purpose library to do Bayesian inference using MCMC. What kabuki provides is a flexible implementation that allows users to easily construct their own hierarchical models. It features a common design where individual subject parameter distributions are themselves draws from a group parameter distribution.

Moreover, kabuki comes equipped with some models of ubiquitous use such as a Bayesian ANOVA model that allows estimation and hypothesis testing of different treatment groups and makes traditional statistical tests such as t and F-tests obsolete. 

Usage
=====

To come...

Getting started
===============

To come...
