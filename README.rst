************
Introduction
************

:Date: July 19, 2011
:Author: Thomas Wiecki, Imri Sofer
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu
:Web site: http://github.com/hddm-dev/kabuki
:Copyright: This document has been placed in the public domain.
:License: Kabuki is released under the GPLv3.
:Version: 0.1.1

Purpose
=======

Kabuki is a Python library intended to make hierarchical PyMC models
reusable, portable and more flexible. Once a model has been formualted
in kabuki it is trivial to apply to apply it to new datasets in
various ways.

Warning
=======

Kabuki is still in early development, documentation is poor to
non-existant. At the moment it is still mostly restricted to the
models we build (e.g. it is the engine for HDDM). If you want to use
kabuki for your own model, We recommend you check back from time to
time to see whether we release something that's actually useful.

Features
========

Kabuki offers (or will offer) the following features for any model you
created in kabuki:

* Easy interface: New model variations can be constructed and
  estimated automatically either via a configuration file and command
  line interface, or from within Python.
* Statistical analysis: Automatically create summary output
  statistics, posterior plots and run poster predictive checks.
* Data generation: By providing a function to generate data from your
  model given paramters you can create new, simulated data sets with
  multiple groups and multiple conditions.
* Model validation: Kabuki will provide automatic tests such as
  parameter recovery from simulated data to validate your model.
* Batteries included: Kabuki will come pre-equipped with some example
  models that are widely used such as a Bayesian ANOVA model for
  hypothesis testing or regression analysis.

Motivation
==========

Hierarchical Bayesian models are gaining popularity in many scientific
disciplines such as cognitive and health sciences, but also
economics. While quite a few useful models have been developed
(e.g. hierarchical Bayesian regression, hierarchical estimation of
drift-diffusion parameters) in the literature, often with reference
implementations in WinBUGS (and sometimes PyMC), applying them to new
data sets requires changing the model code to your specific needs.

If you build your model in kabuki, using it on a new data set with
different conditions and different structure will come for free. All a
user has to do is create a new model object and kabuki will
automatically create a new model tailored to the data in a
user-specified way.

Who is this for?
================

If you are building models in PyMC that you think can be applied to
new data sets with maybe different structure, kabuki will allow you

In psychology, cognitive science but also other fields, a certain hierarchical model structure

that aids creation of hierarchical Bayesian models for cognitive science using PyMC. In many psychological and cognitive experiments multiple subjects are tested on the same task. Traditional non-hierarchical model estimation methods (such as maximum likelihood) either fit a model to each individual subject (ignoring any similarities they have) or to the group as a whole (ignoring individual differences). Due to common scarcety of data the first approach is not feasible in many circumstances. A better model would capture that subjects are different but still similar.

Hierarchical Bayesian modeling allows us to construct models that map the nature of the experiment -- individual subject parameters are themselves distributed according to a group parameter distribution. Kabuki makes it extremely easy to create various models of this structure.

Features
========

* Simplicity: Define 3 functions that return the parameters of your model. The complete model creation is taken care of by kabuki.

* Flexibility in model specification: Kabuki tries not to constrain any models you'll want to create. You specify which parameters your model uses and what distribution the likelihoods have.

* Flexibility in model fitting: After specifiying your model you can tailor it to specific task conditions on the fly (e.g. parameter A in your model depends on the stimulus type that was used).

* Easy specification of models via configuration file fosters exchange of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and goodness-of-fit diagnostics.


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

Getting started
===============
