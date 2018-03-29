Title: NumericalStats: How to randomly sample your empirical arbitrary distribution
Date: 2018-02-18 17:45
Category: MetaAnalysis
Tags: numerical, statistics, python, bayesian
Slug: sampling_arbitrary_distributions_with_python
Cover: /posts/img2104/output_11_0.png
Authors: Peter Schuhmacher
Summary: We provide a simple sampling engine which allows to generate random numbers that are distributed as an empirical and arbitrary distribution given as a data array.

To make as few assumptions as possible is - among other - one motivation to use numerical methods in statistics. If you find some empirical distribution from your problem under consideration, you may be faced with the question how to use this distribution as a **sampling engine**. This is not too difficult, and we give an example here.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
```

### Data generation
We use the beta distribution to generate some arbitrary looking distribution. With different paramters we generate 2 arrays, which we concatenate to one data set of a population. So we have now a distribution that is

- empirical
- discrete (given by data points)
- the analytical function is unknown
- the inverse function is unknown too


```python
prev = 0.7
NP = 10000
nI =  round(NP*prev)
nH = NP-nI

value_h = np.random.beta(2, 9, nH)
value_i = np.random.beta(5, 1, nI)
value_c = np.concatenate((value_h, value_i))
```


```python
pdf, xc = plot_distrib1()
```

![png]({attach}img2104/output_5_0.png)


### Sampling
If we want a random number generator that returns data with the distribution of our empirical distribution we can achieve that in 3 steps

1. we need the cumulative distribution function (**CDF**, also cumulative density function) of our empirical distribution.
2. as driving engine we need from our computer the uniform random generator that gives data in the interval [0, 1], which is the value range of the CDF (and which is the y-axis of the CDF-graph)
3. we have to identify to which element of the CDF the random number fits best and we have to count this hit (this is the transformation to the x-axis)

One can imagine that the uniform random numbers are **sun rays** that are emitted from the y-axis on the left and travel to the right to the CDF-curve. The CDF-curve can be interpreted as a **hill** which get's more **solar energy** on the steeper parts and less on the flater parts due to the inclination. The resulting energy profil will be a data set distributed as the PDF of our empirical distribution.

The CDF can be found as the cumulative sum of our empirical distribution, normalized by the numbers of data.


```python
myPDF = pdf/NP
myCDF = np.cumsum(myPDF)
plot_line()
```

![png]({attach}img2104/output_7_0.png)



### Our random number generator
In the follow code we run with the for-loop _nRuns_ examples and count the hits in the _X_-array. In the code lines inbetween we find out to which data-element of the CDF an emitted unit random number fits best. For that, we pick out the location in the CDF-array where the random number is the first time larger than the CDF-value. Then we round to the closer element.


```python
def run_sampling(myCDF, nRuns=5000):
    X = np.zeros_like(myCDF,dtype=int)
    nX = len(X)-1
    for k in np.arange(nRuns):
        a = random.uniform(0, 1)
        j1 = np.argmax(myCDF>=a)
        if j1==0: j1=1 
        else:  
            eta = (a-myCDF[j1-1])/(myCDF[j1]-myCDF[j1-1])
            j2 = int(j1-1+round(eta))
        X[j2] += 1
    print('nRuns: ',nRuns, '        Sum of X: ', sum(X))
    X[nX] *= 2
    return X/nRuns

X = run_sampling(myCDF)
```

    nRuns:  5000         Sum of X:  5000
    

### The result
As a result we are able to reconstruct the PDF of the empirical distribution with our random number generator using _myCDF_ as input.


```python
plot_distrib2()
```


![png]({attach}img2104/output_11_0.png)

## _Python code: graphics_


```python
def plot_distrib1():
    with plt.style.context('fivethirtyeight'): 
        plt.figure(figsize=(17,5))
        nBins=30
        count_c, bins_c, ignored = plt.hist(value_c, nBins, color='gold', alpha=0.5)      
        dxc = np.diff(bins_c)[0]; xc = bins_c[0:-1] + 0.5*dxc
        plt.plot(xc,count_c, ls='--', lw=1, c='b')
        plt.title('Arbitrary discrete distribution', fontsize=25, fontweight='bold')
        plt.show()
    return count_c,xc
```


```python
def plot_line():
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(17,5))
        plt.plot(xc,myCDF, 'mo-', lw=7, label='discrete CDF', ms=20)
        plt.plot(xc,myPDF, 'co-', lw=7, label='discrete PDF', ms=20)
        plt.xlabel('x-axis'); plt.ylabel('y-axis'); 
        plt.title('CDF and PDF', fontsize=25, fontweight='bold')
        plt.legend(loc='upper center', frameon=False)
        plt.show()
```


```python
def plot_distrib2():
    with plt.style.context('fivethirtyeight'): 
        plt.figure(figsize=(17,5))     
        plt.bar(xc, X ,color='blue', width=0.005, label='resampled PDF')
        plt.plot(xc,myPDF, 'co-', lw=7, label='discrete PDF', ms=20, alpha=0.5)
        plt.title('Reconstruction of the discrete PDF distribution', fontsize=25, fontweight='bold')
        plt.legend(loc='upper center', frameon=False)
        plt.show()
```
