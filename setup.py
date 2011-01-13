from distutils.core import setup
from distutils.extension import Extension
import numpy as np

setup(
    name="kabuki",
    version="0.1",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://code.google.com/p/kabuki",
    packages=["kabuki"],
    description="HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.",
    install_requires=['NumPy >=1.3.0', 'PyMC >=2.0'],
    setup_requires=['NumPy >=1.3.0', 'PyMC >=2.0']
    )

