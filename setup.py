from distutils.core import setup

setup(
    name="kabuki",
    version="0.2RC2",
    author="Thomas V. Wiecki, Imri Sofer",
    author_email="thomas.wiecki@gmail.com",
    url="http://github.com/hddm-devs/kabuki",
    packages=["kabuki"],
    description="kabuki is a python toolbox that allows easy creation of hierarchical bayesian models for the cognitive sciences.",
    install_requires=['NumPy >=1.5.0', 'PyMC >=2.0', 'ordereddict >= 1.1'],
    setup_requires=['NumPy >=1.5.0', 'PyMC >=2.0', 'ordereddict >= 1.1']
)
