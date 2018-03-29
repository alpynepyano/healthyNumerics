Title: NumericalStats: WHY to switch from a decision tree to a Markov model
Date: 2018-03-25 16:50
Category: MetaAnalysis
Tags: numerical, statistics, python, markov model
Slug: from-decision-tree-to-markov-model
Cover: /posts/img2105/output_6_1.png
Authors: Peter Schuhmacher
Summary: We give some arguments, why a change from a decision tree to a Markov model cold be motivated. We provide a code of 7 lines to run a Markov model.


```python
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.colors as mclr
```

## Why a change could be indicated
The **decision tree** is a simple form of decision model. But there are also limitions of the decision tree evident:

- a formal aspect is that the tree format becomes rapidly unwiedly when a combination of several options have to be mapped
- with regard to content the **elapse of time** is not explicit in decision trees. Many chronic diseases such as diabetes, ischaemic heart disease or some forms of cancer have a recuring-remitting pattern over a period of many years. If a longer time horizon has to be adopted, several features may become necessary to be modelled:
    - continuing risk of recurrence
    - competing risk of death as the cohort ages
    - other clinical developments

The **Markov model** is an approach to handel added modelling options. The key structure of a markov model is:

- it is structured around **disease states**
- it is driven by a set of possible transitions between the disease states
- it can be run over a series of time periods which gives an insight over the temporal evolution of the  disease states
- costs may be included in parallel
- the modelled transistions probabilities may change over time too, so that changing conditions may be included


We give in the follwing a basic example of a Markov model that illustrates the temporal evolution of a communicable disease in a population.

$$
\begin{equation}
\begin{array}{rcl}
\textrm{Markov matrix} \; \mathbf{M} &=& \left[\begin{matrix}0.721 & 0.202 & 0.067 & 0.010 \\
                                                            0.000 & 0.581 & 0.407 & 0.012 \\
                                                            0.000 & 0.000 & 0.750 & 0.250 \\                                     
                                                            0.000 & 0.000 & 0.000 & 1.000 \end{matrix}\right] \\
\textrm{Start vector} \;\; \mathbf{p}_0 &=&  \left[\begin{matrix}1 \\ 0 \\0 \\0\end{matrix}\right]\\
Repeat:\\
\textrm{Time step} \;\; \mathbf{p}_1 &=&  \mathbf{M}^T \cdot  \mathbf{p}_0 \\
\textrm{Iteration} \;\; \mathbf{p}_0 &:=& \mathbf{p}_1
\end{array}
\end{equation}
$$




```python
d1
```

![svg]({attach}img2105/output_5_0.svg)

## Set up the Markov system


```python
M_states = np.array(['A (healthy)', 'B (infected)', 'C (ill)', 'D (dead)'])
MM = np.array([[0.721,  0.202,  0.067,  0.010],
               [0.000,  0.581,  0.407,  0.012],
               [0.000,  0.000,  0.750,  0.250],  
               [0.000,  0.000,  0.000,  1.000] ])
dmm = pd.DataFrame(MM, columns=M_states, index=M_states); dmm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A (healthy)</th>
      <th>B (infected)</th>
      <th>C (ill)</th>
      <th>D (dead)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A (healthy)</th>
      <td>0.721</td>
      <td>0.202</td>
      <td>0.067</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>B (infected)</th>
      <td>0.000</td>
      <td>0.581</td>
      <td>0.407</td>
      <td>0.012</td>
    </tr>
    <tr>
      <th>C (ill)</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.750</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>D (dead)</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>



## Run the Markov simulation
We run the simulation with a population of 1000 members. Note that simulation part has 7 lines of code only:


```python
nRuns = 21
m_result = np.zeros((nRuns,len(M_states)))
v0 = np.array([1, 0, 0, 0])*1000
for ir in range(nRuns):
    m_result[ir,:] = v0
    v0 = np.dot(MM.T,v0) 
dmr = pd.DataFrame(np.rint(m_result).astype(int), columns=M_states)
```

## Plot the result


```python
plot_gr02(dmr,'time','# of people','Markov simulation of a communicable disease'); dmr
```

![png]({attach}img2105/output_11_0.png)


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A (healthy)</th>
      <th>B (infected)</th>
      <th>C (ill)</th>
      <th>D (dead)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>721</td>
      <td>202</td>
      <td>67</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>520</td>
      <td>263</td>
      <td>181</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>375</td>
      <td>258</td>
      <td>277</td>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>270</td>
      <td>226</td>
      <td>338</td>
      <td>166</td>
    </tr>
    <tr>
      <th>5</th>
      <td>195</td>
      <td>186</td>
      <td>363</td>
      <td>256</td>
    </tr>
    <tr>
      <th>6</th>
      <td>140</td>
      <td>147</td>
      <td>361</td>
      <td>351</td>
    </tr>
    <tr>
      <th>7</th>
      <td>101</td>
      <td>114</td>
      <td>340</td>
      <td>445</td>
    </tr>
    <tr>
      <th>8</th>
      <td>73</td>
      <td>87</td>
      <td>308</td>
      <td>532</td>
    </tr>
    <tr>
      <th>9</th>
      <td>53</td>
      <td>65</td>
      <td>271</td>
      <td>611</td>
    </tr>
    <tr>
      <th>10</th>
      <td>38</td>
      <td>48</td>
      <td>234</td>
      <td>680</td>
    </tr>
    <tr>
      <th>11</th>
      <td>27</td>
      <td>36</td>
      <td>197</td>
      <td>739</td>
    </tr>
    <tr>
      <th>12</th>
      <td>20</td>
      <td>26</td>
      <td>164</td>
      <td>789</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>19</td>
      <td>135</td>
      <td>831</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10</td>
      <td>14</td>
      <td>110</td>
      <td>865</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7</td>
      <td>10</td>
      <td>89</td>
      <td>893</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>7</td>
      <td>72</td>
      <td>916</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4</td>
      <td>5</td>
      <td>57</td>
      <td>934</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3</td>
      <td>4</td>
      <td>45</td>
      <td>948</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2</td>
      <td>3</td>
      <td>36</td>
      <td>959</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>2</td>
      <td>28</td>
      <td>968</td>
    </tr>
  </tbody>
</table>
</div>




```python
import graphviz as gv
def plotMarkovModel():
    d1 = gv.Digraph(format='png',engine='dot')
    c = ['cornflowerblue','orangered','orange','chartreuse']
    d1.node('A','A (healthy)', style='filled', color=c[0])
    d1.node('B','B (infected)',style='filled', color=c[1])
    d1.node('C','C (ill)',     style='filled', color=c[2])
    d1.node('D','D (dead)',    style='filled', color=c[3])
    d1.edge('A','A',label='0.721'); d1.edge('A','B',label='0.202'); d1.edge('A','C',label='0.067'); d1.edge('A','D',label='0.010')
    d1.edge('B','B',label='0.581');                                 d1.edge('B','C',label='0.407'); d1.edge('B','D',label='0.012')
    d1.edge('C','C',label='0.750');                                                                 d1.edge('C','D',label='0.250')
    d1.edge('D','D',label='1.000');
    return d1
d1 = plotMarkovModel()
d1.render('img/mamo1', view=True)

def plot_gr02(DF,xLabel,yLabel,grTitel):
    with plt.style.context('fivethirtyeight'): 
         fig = plt.figure(figsize=(15,7)) ;
         ax1 = fig.add_subplot(111);
         #DF.plot(ax = plt.gca())
         colors = ['cornflowerblue','orangered','orange','chartreuse']
         DF.plot(ax = ax1, style='-', color=colors)
         plt.xlabel(xLabel);  plt.ylabel(yLabel);
         plt.title(grTitel, fontsize=25, fontweight='bold');
         plt.show()
```
