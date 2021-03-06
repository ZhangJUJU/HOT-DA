# Hierarchical Optimal Transport for Unsupervised Domain Adaptation (HOT-DA) - Submitted to AAAI22

## Installation and Dependencies

**HOT-DA** is based on `NumPy`, `Pandas`, `Scikit-Learn` and `POT`. 
So, make sure these packages are installed. For example, you can install them with `pip`:

```
pip3 install numpy pandas POT
```

It is recommend to use `Python 3.8` from [Anaconda](https://www.anaconda.com/) distribution. All the codes for the article are tested on macOS Big Sur Version 11.2.3


## Scripts for experiments:
To reproduce the results, four `jupyter notebook` are provided:

 1 - `/Moons/Experimentation_Moons.ipynb`
 
 2.a - `/Digits/Mnist_USPS.ipynb`
 
 2.b - `/Digits/USPS_Mnist.ipynb`
 
 3 - `/Objetcs/Experimentation_Objects.ipynb`
 
 We also release the used `datasets` (within each of the three files: Moons, Digits and Objects)
 
 
 ## Scripts for figures:

● Comparaison with structure-target-agnostic methods - `/Comparaison with structure-target-agnostic methods.ipynb`

● Wasserstein-Spectral clustering - `/Wasserstein Spectral Clustering.ipynb`

● Decision boundary on moons dataset - `/Moons_DecisionBoundary.ipynb`

● Unbalanced structures analysis - `/Unbalanced structures sensitivity.ipynb`


## Other scripts:

● Main class for our algorithm:  `/HOTDA.py`
