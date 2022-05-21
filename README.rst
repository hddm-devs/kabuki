************
Introduction
************

:Author: Thomas Wiecki, Imri Sofer
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu
:Web site: http://github.com/hddm-dev/kabuki
:Copyright: This document has been placed in the public domain.
:License: Simplified BSD (see LICENSE)
:Version: 0.6.5

Purpose
=======

Kabuki is a Python library intended to make hierarchical PyMC models
reusable, portable and more flexible. Once a model has been formulated
in kabuki it is trivial to apply it to new datasets in various
ways. Currently, it is geared towards hierarchical Bayesian models
that are common in the cognitive sciences but it might be easily
adapted to other domains.

In essence, kabuki allows easy creation of model-factories. After
specifiyng the model structure, models tailored to new data sets and
new configurations can be instantiated easily.

To see it in action, check out HDDM_ which uses kabuki for the heavy
lifting. Especially the How-to_ should give a comprehensive overview
of kabuki's features.

Features
========

* Easy model specification: It is quite trivial to convert an existing
  PyMC model to kabuki.
* Models are classes: The resulting kabuki model is one class with
  methods for setting the nodes to their MAP, sampling from the
  posterior, saving and loading models, plotting output statistics
  etc.
* Easy interface: New model variations can be constructed and
  estimated automatically either.
* Statistical analysis: Automatically create nicely formatted summary
  output statistics, posterior plots and run posterior predictive checks.
* Data generation: By providing a function to generate data from your
  model given paramters you can create new, simulated data sets with
  multiple groups and multiple conditions (e.g. for testing parameter
  recovery).
* Batteries included: Over time we will add more standard models to
  kabuki such as a model to perform Bayesian ANOVA or regression
  analysis.

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
user has to do is instantiate the model class and kabuki will
automatically create a new PyMC model tailored to the data in a
user-specified way.

Usage
=====

Since kabuki builds on top of PyMC you have to know the basic model
creation process there. Check out the `PyMC documentation`_ first if
you are not familiar.

To create your own model you have to inherit from the
`kabuki.Hierarchical` base class which provides all of the
functionality. Instead of directly instantiating PyMC nodes you have
to wrap them in a kabuki.Knode object. This provides the information
of how to create PyMC nodes once the model is instantiated.

Example model
-------------

Here is a very simple example model where we assume that multiple
subjects provided normal distributed data and we want to infer the
mean of each subject but also assume that subject means themselves are
distributed according to a normal group distribution for which we
want to estimate the mean and variance:

::

    from kabuki import Hierarchical, Knode
    import pymc

    class MyModel(Hierarchical):
        # We have to overload the create_knodes() method. It is
        # expected to return a list of knodes.
        def create_knodes(self):

	    mu_g = Knode(pymc.Uniform, 'mu_g', lower=-5, upper=5, depends=self.depends['mu'])

            mu_subj = Knode(pymc.Normal, 'mu_subj', mu=mu_g, tau=1, depends=('subj_idx',), subj=True)

            like = Knode(pymc.Normal, 'like', mu=mu_subj, tau=1, col_name='data', observed=True)

            return [mu_g, mu_subj, like]

OK, what's going on here?

Creation of group mu node
"""""""""""""""""""""""""

The first line of create_knodes() creates the group mean knode.

* The first argument is the pymc distribution of the parameter.

* The second argument is the name you want to give to this knode
  'lower' and 'upper' in this case are keyword arguments that get
  passed to PyMC during node creation.

* The `depends` keyword argument means that seperate PyMC nodes can be
  created for user-supplied conditions (this will become clear later).

* `self.depends` is a user-supplied dictionary that maps a parameter
  name to a column in the data specifying the different
  conditions. Kabuki will then create different group mean nodes
  depending on the conditions found in this data column.

Creation of subject node
""""""""""""""""""""""""
In the second line we create the subject knode with `mu` being
distributed according to the parent distribution we just created.
Linking the hierarchical structure works in the same way as with PyMC
nodes, however, note that here we pass in our newly create Knode to `mu`.

Note moreover that while we say that this node depends on the data
column 'subj_idx' (this column is supposed to have all the subject
indices), we don't have to specify again that this child node also
depends on the user-specified column.  Kabuki knows that since the
parent (mu_group) depends on a user-defined column name, the child
(mu_subj) also has to depend on the same conditions in the data.

The subj keyword specifies that this is a subject knode (this is
required for internal purposes).

Creation of observed node
"""""""""""""""""""""""""

Finally, we have to create the likelihood or observed node.  The only
difference to before is the observed=True keyword and col_name which
specifies on which data column the likelihood depends on. As we will
see later, kabuki will parcel the data column appropriately so that
each subject observed node is linked to the data belonging to that
subject (and that condition).

Running the example model
-------------------------

After we specified our model in this way we can construct new models
very easily. Say we had an experiment where we tested each subject on
two conditions, 'low' and 'high', and we suspect that this will result
in different means of their normal distributed responses.

An example data file in csv might looks this:

::

    subj_idx, data, condition
    1,        0.3,  'low'
    1,        -0.25,'low'
    1,        1.3,  'high'
    1,        0.5.1.dev,  'high'
    [...]
    24,       0.8,  'low'
    24,       0.1,  'high'

Here is how you would create a model tailored around this data set,
set the parameters to their MAP, sample and print some output statistics:

::

   data = kabuki.load_csv('data.csv')
   # create the model. depends_on tells it that the parameter
   # 'mu' (this links to depends=self.depends['mu'] we specified above
   # when we created the group knode) depends on the data column
   # 'condition'
   model = MyModel(data, depends_on={'mu': 'condition})

   model.map()
   model.sample(5000, burn=1000)

   # Print the stats to the console
   model.print_stats()
   # Plot posterior distributions
   model.plot_posteriors()
   # Plot the posterior predictive on top of the subject data
   model.plot_posterior_predictive()

Conclusion
----------

The resulting model will have 2 group-mean distributions ('mu_low' and
'mu_high', one for each condition), 2 subject-mean distributions per
subject (so 48 in total, assuming we had 24 subjects, which are linked
to their appropriate group-mean) and 2 likelihoods (i.e. observeds)
per subject which are linked to the appropriate subject's data.

As you can see, kabuki takes care of creating multiple nodes where
appropriate (i.e. for different conditions), provides meaningful names
and parcels the data so that the likelihoods are linked correctly.

There are many more features for more complex models and advanced
diagnostics (like posterior predictive checks).

.. _PyMC documentation: http://pymc-devs.github.com/pymc/
.. _HDDM: https://github.com/hddm-devs/hddm/
.. _How-to: http://ski.clps.brown.edu/hddm_docs/howto.html
