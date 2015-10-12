import warnings
import numpy as np
import pymc as pm

from pymc import Stochastic, utils
from pymc.distributions import bind_size

# To remain compatibility with pymc 2.1, we copy this function.
# Once pymc 2.2 is released this should be depracated.

def debug_wrapper(func, name):
    # Wrapper to debug distributions

    import pdb

    def wrapper(*args, **kwargs):

        print(('Debugging inside %s:' % name))
        print('\tPress \'s\' to step into function for debugging')
        print('\tCall \'args\' to list function arguments')

        # Set debugging trace
        pdb.set_trace()

        # Call function
        return func(*args, **kwargs)

    return wrapper


def new_dist_class(*new_class_args):
    """
    Returns a new class from a distribution.

    :Parameters:
      dtype : numpy dtype
        The dtype values of instances of this class.
      name : string
        Name of the new class.
      parent_names : list of strings
        The labels of the parents of this class.
      parents_default : list
        The default values of parents.
      docstr : string
        The docstring of this class.
      logp : function
        The log-probability function for this class.
      random : function
        The random function for this class.
      mv : boolean
        A flag indicating whether this class represents array-valued
        variables.

      .. note::
        stochastic_from_dist provides a higher-level version.

      :SeeAlso:
        stochastic_from_dist
    """

    (dtype, name, parent_names, parents_default, docstr, logp, random, mv, logp_partial_gradients) = new_class_args

    class new_class(Stochastic):
        __doc__ = docstr

        def __init__(self, *args, **kwds):
            (dtype, name, parent_names, parents_default, docstr, logp, random, mv, logp_partial_gradients) = new_class_args
            parents=parents_default

            # Figure out what argument names are needed.
            arg_keys = ['name', 'parents', 'value', 'observed', 'size', 'trace', 'rseed', 'doc', 'debug', 'plot', 'verbose']
            arg_vals = [None, parents, None, False, None, True, True, None, False, None, -1]
            if 'isdata' in kwds:
                warnings.warn('"isdata" is deprecated, please use "observed" instead.')
                kwds['observed'] = kwds['isdata']
                pass


            # No size argument allowed for multivariate distributions.
            if mv:
                arg_keys.pop(4)
                arg_vals.pop(4)

            arg_dict_out = dict(list(zip(arg_keys, arg_vals)))
            args_needed = ['name'] + parent_names + arg_keys[2:]

            # Sort positional arguments
            for i in range(len(args)):
                try:
                    k = args_needed.pop(0)
                    if k in parent_names:
                        parents[k] = args[i]
                    else:
                        arg_dict_out[k] = args[i]
                except:
                    raise ValueError('Too many positional arguments provided. Arguments for class ' + self.__class__.__name__ + ' are: ' + str(args_needed))


            # Sort keyword arguments
            for k in args_needed:
                if k in parent_names:
                    try:
                        parents[k] = kwds.pop(k)
                    except:
                        if k in parents_default:
                            parents[k] = parents_default[k]
                        else:
                            raise ValueError('No value given for parent ' + k)
                elif k in list(arg_dict_out.keys()):
                    try:
                        arg_dict_out[k] = kwds.pop(k)
                    except:
                        pass

            # Remaining unrecognized arguments raise an error.
            if len(kwds) > 0:
                raise TypeError('Keywords '+ str(list(kwds.keys())) + ' not recognized. Arguments recognized are ' + str(args_needed))

        # Determine size desired for scalar variables.
        # Notes
        # -----
        # Case | init_val     | parents       | size | value.shape | bind size
        # ------------------------------------------------------------------
        # 1.1  | None         | scalars       | None | 1           | 1
        # 1.2  | None         | scalars       | n    | n           | n
        # 1.3  | None         | n             | None | n           | 1
        # 1.4  | None         | n             | n(m) | n (Error)   | 1 (-)
        # 2.1  | scalar       | scalars       | None | 1           | 1
        # 2.2  | scalar       | scalars       | n    | n           | n
        # 2.3  | scalar       | n             | None | n           | 1
        # 2.4  | scalar       | n             | n(m) | n (Error)   | 1 (-)
        # 3.1  | n            | scalars       | None | n           | n
        # 3.2  | n            | scalars       | n(m) | n (Error)   | n (-)
        # 3.3  | n            | n             | None | n           | 1
        # 3.4  | n            | n             | n(m) | n (Error)   | 1 (-)

            if not mv:

                shape = arg_dict_out.pop('size')
                shape = None if shape is None else tuple(np.atleast_1d(shape))

                init_val = arg_dict_out['value']
                init_val_shape = None if init_val is None else np.shape(init_val)

                if len(parents) > 0:
                    pv = [np.shape(utils.value(v)) for v in list(parents.values())]
                    biggest_parent = np.argmax([(np.prod(v) if v else 0) for v in pv])
                    parents_shape = pv[biggest_parent]

                    # Scalar parents can support any shape.
                    if np.prod(parents_shape) < 1:
                        parents_shape = None

                else:
                    parents_shape = None

                def shape_error():
                    raise ValueError('Shapes are incompatible: value %s, largest parent %s, shape argument %s'%(shape, init_val_shape, parents_shape))

                if init_val_shape is not None and shape is not None and init_val_shape != shape:
                    shape_error()

                given_shape = init_val_shape or shape
                bindshape = given_shape or parents_shape

                # Check consistency of bindshape and parents_shape
                if parents_shape is not None:
                    # Uncomment to leave broadcasting completely up to NumPy's random functions
                    # if bindshape[-np.alen(parents_shape):]!=parents_shape:
                    # Uncomment to limit broadcasting flexibility to what the Fortran likelihoods can handle.
                    if bindshape<parents_shape:
                        shape_error()

                if random is not None:
                    random = bind_size(random, bindshape)


            elif 'size' in list(kwds.keys()):
                raise ValueError('No size argument allowed for multivariate stochastic variables.')


            # Call base class initialization method
            if arg_dict_out.pop('debug'):
                logp = debug_wrapper(logp)
                random = debug_wrapper(random)
            else:
                Stochastic.__init__(self, logp=logp, random=random, dtype=dtype, **arg_dict_out)

    new_class.__name__ = name
    new_class.parent_names = parent_names
    new_class.parents_default = parents_default
    new_class.dtype = dtype
    new_class.mv = mv
    new_class.raw_fns = {'logp': logp, 'random': random}

    return new_class

def scipy_stochastic(scipy_dist, **kwargs):
    """
    Return a Stochastic subclass made from a particular SciPy distribution.
    """
    import inspect
    import scipy.stats.distributions as sc_dst
    from pymc.ScipyDistributions import separate_shape_args

    if scipy_dist.__class__.__name__.find('_gen'):
        scipy_dist = scipy_dist(**kwargs)

    name = scipy_dist.__class__.__name__.replace('_gen','').capitalize()

    (args, varargs, varkw, defaults) = inspect.getargspec(scipy_dist._pdf)

    shape_args = args[2:]
    if isinstance(scipy_dist, sc_dst.rv_continuous):
        dtype=float

        def logp(value, **kwds):
            args, zkwds = separate_shape_args(kwds, shape_args)
            if hasattr(scipy_dist, '_logp'):
                return scipy_dist._logp(value, *args)
            else:
                return np.sum(scipy_dist.logpdf(value,*args,**kwds))

        parent_names = shape_args + ['loc', 'scale']
        defaults = [None] * (len(parent_names)-2) + [0., 1.]

    elif isinstance(scipy_dist, sc_dst.rv_discrete):
        dtype=int

        def logp(value, **kwds):
            args, kwds = separate_shape_args(kwds, shape_args)
            if hasattr(scipy_dist, '_logp'):
                return scipy_dist._logp(value, *args)
            else:
                return np.sum(scipy_dist.logpmf(value,*args,**kwds))

        parent_names = shape_args + ['loc']
        defaults = [None] * (len(parent_names)-1) + [0]
    else:
        return None

    parents_default = dict(list(zip(parent_names, defaults)))

    def random(shape=None, **kwds):
        args, kwds = separate_shape_args(kwds, shape_args)

        if shape is None:
            return scipy_dist.rvs(*args, **kwds)
        else:
            return np.reshape(scipy_dist.rvs(*args, **kwds), shape)

    # Build docstring from distribution
    docstr = name[0]+' = '+name + '(name, '+', '.join(parent_names)+', value=None, shape=None, trace=True, rseed=True, doc=None)\n\n'
    docstr += 'Stochastic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += """
Methods:

    random()
        - draws random value
          sets value to return value

    ppf(q)
        - percent point function (inverse of cdf --- percentiles)
          sets value to return value

    isf(q)
        - inverse survival function (inverse of sf)
          sets value to return value

    stats(moments='mv')
        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')


Attributes:

    logp
        - sum(log(pdf())) or sum(log(pmf()))

    cdf
        - cumulative distribution function

    sf
        - survival function (1-cdf --- sometimes more accurate)

    entropy
        - (differential) entropy of the RV.


NOTE: If you encounter difficulties with this object, please try the analogous
computation using the rv objects in scipy.stats.distributions directly before
reporting the bug.
    """

    new_class = new_dist_class(dtype, name, parent_names, parents_default, docstr, logp, random, True, None)
    class newer_class(new_class):
        __doc__ = docstr
        rv = scipy_dist
        rv.random = random

        def __init__(self, *args, **kwds):
            new_class.__init__(self, *args, **kwds)
            self.args, self.kwds = separate_shape_args(self.parents, shape_args)
            self.frozen_rv = self.rv(self.args, self.kwds)
            self._random = bind_size(self._random, self.shape)

        def _pymc_dists_to_value(self, args):
            """Replace arguments that are a pymc.Node with their value."""
            # This is needed because the scipy rv function transforms
            # every input argument which causes new pymc lambda
            # functions to be generated. Thus, when calling this many
            # many times, excessive amounts of RAM are used.
            new_args = []
            for arg in args:
                if isinstance(arg, pm.Node):
                    new_args.append(arg.value)
                else:
                    new_args.append(arg)

            return new_args

        def pdf(self, value=None):
            """
            The probability distribution function of self conditional on parents
            evaluated at self's current value
            """
            if value is None:
                value = self.value
            return self.rv.pdf(value, *self._pymc_dists_to_value(self.args), **self.kwds)

        def cdf(self, value=None):
            """
            The cumulative distribution function of self conditional on parents
            evaluated at self's current value
            """
            if value is None:
                value = self.value
            return self.rv.cdf(value, *self._pymc_dists_to_value(self.args), **self.kwds)

        def sf(self, value=None):
            """
            The survival function of self conditional on parents
            evaluated at self's current value
            """
            if value is None:
                value = self.value
            return self.rv.sf(self.value, *self._pymc_dists_to_value(self.args), **self.kwds)

        def ppf(self, q):
            """
            The percentile point function (inverse cdf) of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.ppf(q, *self._pymc_dists_to_value(self.args), **self.kwds)
            return self.value

        def isf(self, q):
            """
            The inverse survival function of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.isf(q, *self._pymc_dists_to_value(self.args), **self.kwds)
            return self.value

        def stats(self, moments='mv'):
            """The first few moments of self's distribution conditional on parents"""
            return self.rv.stats(moments=moments, *self._pymc_dists_to_value(self.args), **self.kwds)

        def _entropy(self):
            """The entropy of self's distribution conditional on its parents"""
            return self.rv.entropy(*self._pymc_dists_to_value(self.args), **self.kwds)
        entropy = property(_entropy, doc=_entropy.__doc__)

    newer_class.__name__ = new_class.__name__
    return newer_class

