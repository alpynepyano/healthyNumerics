Title: Numerical Stats 02: π by Monte Carlo integration
Date: 2017-08-12 21:30
Category: MetaAnalysis
Tags: Monte Carlo, numerical analysis, Python
Slug: numerical_stats_02_monte_carlo_integration
Cover: /posts/img2102/output_4_1.png
Authors: Peter Schuhmacher
Summary: We use a Monte Carlo method with a code of 6 lines for the integration of mathmatical functions. In the case of a circle we can determin π.

## Introduction
**Monte Carlo methods** are probablistic computational techniques. In the core a Monte Carlo algorithm depicts randomly certain values from the value space of a parameter under consideration.  Combining several parameters allows to draw stochastic conclusions of relationships. The integration of mathematical functions of the form
$$
A = \int_a^b f(x) \cdot dx
$$
gives a certain insight how the Monte Carlo method works. 

In oder to perform an integration we want to know how the randomly selected values are distributed: which of the values are equal or smaller than the value of the function and which ones are greater. This is a binary decision that divides the random values in two groups. From the ratio of the size of the groups we can draw our conclusions.

We use the function (= integrand) as a decision criteria only. The algorithm delivers us nothing else than counts/frequencies. The probalistic closure is then:

$$
\frac{\textrm{favourable cases}} {\textrm{possible cases}} = \frac {n}{N} = \frac{A_{below\;function}}{A_{total\;area}}
$$

The area $A_{below\;function}$ is the unknown area we are interested in. For $A_{total\;area}$ we choose arbitrary  a simple region those area we can compute without difficulties. 

## Example: determination of $\pi$
As an example we choose a circle whose mathematical function is given by the first or the second line:

$$
x^2 + y^2 = R^2 \\
y = f(x) = \sqrt{R^2 - x^2}
$$

For the estimate of $\pi$ the area of the circle is compared with the area of the square 2R by 2R, This ratio is $\pi/4$: 
$$
A_{circle} = R^2 \cdot \pi \\
A_{square} = {(2R)}^2 = 4R^2
$$

We randomly generate $(x_p,y_p)$-points. For each point we have to decide wether it is inside or outside the circle. For that we use the difference $y_p - f(x)$, where  $f(x) = \sqrt{R^2 - x^2}$ is the function for a circle in the first quadrant. We can count how many points are inside the circle. The ratio $n/N$ is assumed to be equal to the ratio $A_{circle}/A_{square}$

$$
\frac {\textrm{n = (x,y)-points in the circle}}  {\textrm{N = all (x,y)-points in the square}} = \frac{A_{circle}}{A_{square}}
= \frac {R^2 \cdot \pi}{4R^2 } = \frac {\pi}{4} \\
\pi = 4\frac{n}{N}
$$


## Python code

In contrast to the explanation, the python code needs **just 6 lines**. The number of trials N should be choosen rather 100'000 than 10'000 (which we used to make the grafical display "readeble")


```python
import numpy as np
import matplotlib.pyplot as plt

def f(x): return np.sqrt(1-x*x)  # function for a circle

N = 10000   # number of trials
np.random.seed(2)
x = np.random.rand(N)
y = np.random.rand(N)
n = np.sum( y - f(x) < 0.0) #number of points in the circle


#----- Output and Graphics -------------------
print('PI numpy       : ', np.pi)
print('PI monte carlo : ', 4*n/N)
print('difference     : ', 4*n/(N) - np.pi)

xp = np.linspace(0,1,50)
colors = (np.sign(f(x)-y)+1)/2
area = 10
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplot(111)
plt.plot(xp,f(xp),'--k')
plt.fill_between(xp, f(xp), color='darkblue', alpha=0.5 )
plt.scatter(x, y, s=area, c=colors, alpha=0.9)
plt.axis('equal')
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()
```

    PI numpy       :  3.141592653589793
    PI monte carlo :  3.1436
    difference     :  0.00200734641021
    


![png]({attach}img2102/output_4_1.png)


```python

```
