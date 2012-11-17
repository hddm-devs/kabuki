from distutils.core import setup

setup(
    name="kabuki",
    version="0.4RC2",
    author="Thomas V. Wiecki, Imri Sofer",
    author_email="thomas.wiecki@gmail.com",
    url="http://github.com/hddm-devs/kabuki",
    packages=["kabuki"],
    description="kabuki is a python toolbox that allows easy creation of hierarchical bayesian models for the cognitive sciences.",
    install_requires=['NumPy >= 1.5.0', 'PyMC >= 2.2', 'ordereddict >= 1.1', 'pandas >= 0.6.0', 'matplotlib >= 1.0.0'],
    setup_requires=['NumPy >= 1.5.0', 'PyMC >= 2.2', 'ordereddict >= 1.1', 'pandas >= 0.6.0', 'matplotlib >= 1.0.0']
)
