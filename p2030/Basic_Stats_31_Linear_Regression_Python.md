Title: Basic Stats 31: Linear Regression
Date: 2017-04-26 08:30
Category: MetaAnalysis
Tags: Python, Statistics
Slug: Basic_Stats_31_Linear_Regression_Python
Cover: /posts/img2030/output_7_0.png
Authors: Peter Schuhmacher
Summary: Elementary methods for data flow and statistics with Python, Pandas, NumPy, StatsModels, Seaborn, Matplotlib

###  Einfache lineare Regression

Einfache Methoden für Datenfluss und Statistik mit **Python, Pandas, NumPy, StatsModels, Seaborn, Matplotlib**

Das Beispiel stammt aus:

>Reinhold Hatzinger, Kurt Hornik, Herbert Nagel (2011): **R - Einführung durch angewandte Statistik**, Pearson Studium, 465pp, ISBN978-3-8632-6599-1 , siehe auch [hier](https://www.pearson-studium.de/r.html)

Dort könnten auch unter Extras/CWS die Input-Daten gefunden werden. Das nachfolgende Beispiel ist aus **Kapitel 9.2** (Mehrere metrische Variablen), die verwendete Datendatei: gewicht.csv.

Im Beispiel liegen Daten zum Körpergewicht vor, das einerseits erfragt und anderseits mittels Waage gemessen wurde.


```python
import numpy  as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
%matplotlib inline
```

### Daten
    angabe;waage
    83;88,0
    64;70,1
    94;94,2
    73;76,8
    79;81,1

Die Daten liegen in einer CSV-Datei vor. Oben sind nur die ersten paar Zeilen davon angezeigt. Sie können mit pd.read_csv direkt in ein **Pandas** DataFrame eingelesen werden. Dabei muss mitgeteilt werden, dass ";" als Trennzeichen auftritt. In der zweiten Spalte müssen die "," durch "." ersetzt werden. Da die zweite Spalte danach als string vorliegt, wird sie mit pd.to_numeric in einen numerischen Wert umgewandelt.


```python
f10Dir = r"C:/gcg/7_Wissen_T/eBücher/R-HatzingerHornikNagel/R-Begleitmaterial/Daten\\"
f10Name= r"gewicht.csv"
f10 = f10Dir+f10Name

dg = pd.read_csv(open(f10, newline=''), sep=';') # set ; as delimiter and import the data as pd.DataFrame
dg['waage'] = dg['waage'].str.replace(",",".")   # replace the coma by a point
dg['waage'] = pd.to_numeric(dg['waage'])         # transform the string into numerical value

dg.head(4)                                       # display the first few lines of the DataFrame
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>angabe</th>
      <th>waage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>83</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>64</td>
      <td>70.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>94</td>
      <td>94.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73</td>
      <td>76.8</td>
    </tr>
  </tbody>
</table>
</div>



### Darstellung der Daten
Mit **Seaborn** können die Daten dargestellt werden. Die Regressionsgerade sowie die Häufigkeitsverteilungen werden automatisch berechnet und geplottet.


```python
sns.jointplot(x="waage", y="angabe", data= dg, kind="reg");
```


![png]({attach}img2030/output_7_0.png)


### Parameter der Regressionsgerade
Wir berechnen nun die Regression mit **StatsModels**. Die gewünschte Regressionsbeziehung kann mit *formula='angabe ~ waage '* in einem Format angegeben werden, wie es bei **R** üblich ist: y=angabe soll durch x=waage ausgedrückt werden. Die Ergebnisparameter liegen als pd.DataSeries vor, also ein 1-dimensionales DataFrame.


```python
estimation = smf.ols(formula='angabe ~ waage ', data=dg).fit()
p = estimation.params
print(type(p))
print(p)
```

    <class 'pandas.core.series.Series'>
    Intercept   -3.522249
    waage        1.015543
    dtype: float64
    

### Verwendung der Regressionsgerade
Mit den Regressionsparametern drücken wir die Regressionsgerade aus. Für die x-Werte bilden wir einen **NumPy**-array, und die y-Werte berechnen wir mit der Geradengleichung. Mit **Matplotlib** plotten wir die Daten und die Gerade.


```python
x = np.linspace(0,120,2)
y = p[0] + p[1]*x
fig,ax = plt.subplots()
ax.plot(x,y,'-')
ax.plot(dg['waage'],dg['angabe'],'o')
```





![png]({attach}img2030/output_11_1.png)


Von den StatsModels-Ergebnissen drucken wir die vollständige Zusammenfassung aus.


```python
estimation.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>angabe</td>      <th>  R-squared:         </th> <td>   0.970</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.969</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1529.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 12 Sep 2017</td> <th>  Prob (F-statistic):</th> <td>4.62e-38</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:14:35</td>     <th>  Log-Likelihood:    </th> <td> -112.78</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   229.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    48</td>      <th>  BIC:               </th> <td>   233.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -3.5222</td> <td>    2.132</td> <td>   -1.652</td> <td> 0.105</td> <td>   -7.808</td> <td>    0.764</td>
</tr>
<tr>
  <th>waage</th>     <td>    1.0155</td> <td>    0.026</td> <td>   39.107</td> <td> 0.000</td> <td>    0.963</td> <td>    1.068</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.153</td> <th>  Durbin-Watson:     </th> <td>   1.858</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.926</td> <th>  Jarque-Bera (JB):  </th> <td>   0.318</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.105</td> <th>  Prob(JB):          </th> <td>   0.853</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.671</td> <th>  Cond. No.          </th> <td>    525.</td>
</tr>
</table>



## Lineare Regression mit scipy


```python
slope, intercept, r_value, p_value, std_err = st.linregress(dg['waage'], dg['angabe'])
print('scipy lingress: ',slope, intercept)
```

    scipy lingress:  1.01554288925 -3.52224854645
    

## Lineare Regression mit numpy


```python
f_poly = np.polyfit(dg['waage'], dg['angabe'], 1)
print('numpy polyfit  : ', f_poly)
```

    numpy polyfit  :  [ 1.01554289 -3.52224855]
    

## Function fitting with scipy


```python
from scipy.optimize import curve_fit
def func(x, a, b):  return a * x + b
popt, pcov = curve_fit(func, dg['waage'], dg['angabe'])
print(popt)
```

    [ 1.01554289 -3.52224855]
    


```python

``````
