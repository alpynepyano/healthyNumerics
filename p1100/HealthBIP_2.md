Title: "America first, Switzerland second" - geschafft !
Date: 2018-02-06 16:30
Category: HealthPoliticsEconomics 
Tags: health, costs, Switzerland, USA, OECD
Slug: cost-healthcaresystem-usa-switzerland
Cover: /posts/img1100/output_2_0.png
Authors: Peter Schuhmacher
Summary: Die OECD-Daten zeigen, dass das US-Gesundheitssystem nicht mit dem Rest der Welt mithalten kann. Es ist das teuerste System, gefolgt von der Schweiz.

Die OECD-Publikation 2017 bringt es erneut an den Tag: bei den Gesundheitsausgaben gemessen am Inlandbruttosozialprodukt steht

- an erster Stelle: die USA
- an zweiter Stelle: die Schweiz


```python
plot_HEGDP()
```

![png]({attach}img1100/output_2_0.png)


# Die Jahre davor

Blickt man die Jahre zurück, so war das früher auch schon so, aber von 2005 bis 2011 gab es doch eine Periode, in welcher der naturgesetzlich anmutende Zusammenhang durchbrochen wurde.

Die nachstehende Tabelle zeigt **den Rang** an, den die USA und die Schweiz im Felde der 45 Länder einnehmen, über die der OECD-Bericht berichtet. **GDP** steht für den Anteil der Gesundheitsausgaben gemessen am Inlandbruttosozialprodukt (GDP), und **LE** steht für **Lebenserwartung**


```python
dr
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
      <th>Jahr</th>
      <th>Daten</th>
      <th>GDP USA</th>
      <th>GDP CH</th>
      <th>LE USA</th>
      <th>LE CH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>2016</td>
      <td>1</td>
      <td>2</td>
      <td>28</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>2013</td>
      <td>1</td>
      <td>2</td>
      <td>28</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013</td>
      <td>2011</td>
      <td>1</td>
      <td>6</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>2009</td>
      <td>1</td>
      <td>7</td>
      <td>27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>2007</td>
      <td>1</td>
      <td>3</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2007</td>
      <td>2005</td>
      <td>1</td>
      <td>4</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005</td>
      <td>2003</td>
      <td>1</td>
      <td>2</td>
      <td>22</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2003</td>
      <td>2001</td>
      <td>1</td>
      <td>2</td>
      <td>21</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2001</td>
      <td>1998</td>
      <td>1</td>
      <td>2</td>
      <td>19</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Die Veränderung des Ranges zeigt an, wie ein einzelnes Land gewisse Entwicklungen im Vergleich zu den andern Ländern mitmacht.

Die USA ist mit Dauerrang 1 nicht nur das stetig teuerste Gesundheitswesen der Welt. **Gleichzeitig kann die USA  mit seinen Entwicklungen im Gesundheitswesen mit dem Rest der OECD-Welt nicht mithalten**. In den fast 20 Jahren, die in OECD-Berichten vergleichend rapportiert sind, hat die USA bezüglich der Lebenserwartung stetig an Rang verloren, nämlich von Rang 19 rückwärts zu Rang 28. Das ist eine schlechte Bilanz !

Für die Schweiz ist die Situation auch nicht zum die Hände in den Schoss legen. Aber immerhin liegt die Schweiz bezüglich der Lebenserwartung regelmässig im Breich der drei Spitzenplätze weltweit. Das beruhigt. Würde man hierzulande statt nur  Mittelwerte die statistische Verteilung des noch frei verfügbaren Einkommens der privaten Haushalte auch ansehen, dann wäre etwas Beunruhigung allerdings durchaus sachgerecht. 


```python
plot_LE()
```

![png]({attach}img1100/output_6_0.png)



# Wo bleibt die Evidenz ökonomischer Thesen ?

Das Gesundheitswesen der USA gilt als dasjenige mit dem höchsten Grad an Privatisierung und Dergulierung. Privatisierung und Dergulierung sind die beiden Rezepte, die auch in der Scheiz regelmässig zur Effizienzsteigerung vorgebracht werden. In Anbetracht der OECD-Datenreihe **wirft das schon Fragen zur Evidenz ökonomischer, resp. politischer Thesen auf**:

> Wenn wir in der Schweiz das zweitteuerste Gesundheitswesen der Welt haben, ist es dann richtig, aus den USA die Rezepte des allerteuersten Gesundheitswesen dieser Erde zu importieren, das zudem mit dem Rest der Welt nicht mithalten kann ?

Meines Erachtens gibt es weitere Topics, die nicht nur einem liberalen Staatswesen gut anstehen, sondern die gar als Voraussetzung für das Funktionieren marktorientierter Systeme gelten. Darunter gehören unter anderem:

- good political governance (auf dass das _check and balances_ in unserem ausgeklügelten politischen System wirklich stattfindet)
- Transparenz
- Nutzung der Daten

In diesen Punkten sündigen wir in der Schweiz mitunter gänzlich ungeniert.


```python
plot_HEGDP_sort()
```

    C:\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:545: UserWarning: No labelled objects found. Use label='...' kwarg on individual plots.
      warnings.warn("No labelled objects found. "
    

![png]({attach}img1100/output_8_1.png)


**Grafik**

Der staatliche Finanzierungsnateil ist innerhalb von Europa in der Schweiz fast am geringsten. Aus dieser Perspektive ist das Gesundheitssystem der Schweiz zu einem hohen Anteil "privatisiert", indem die Patienten privat _out-of-the-pocket_ einen hohen Anteil bezahlen.








## Quelle

http://www.oecd-ilibrary.org/social-issues-migration-health/health-at-a-glance-2017_health_glance-2017-en

OECD (2017), Health at a Glance 2017: OECD Indicators, OECD Publishing, Paris.
http://dx.doi.org/10.1787/health_glance-2017-en





## ANHANG: Python Code


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
```


```python
d = {   'Jahr': [2017, 2015, 2013, 2011, 2009, 2007, 2005, 2003, 2001], 
       'Daten': [2016, 2013, 2011, 2009, 2007, 2005, 2003, 2001, 1998], 
     'GDP USA': [1, 1, 1, 1, 1, 1, 1, 1, 1], 
      'GDP CH': [2, 2, 6, 7, 3, 4, 2, 2, 2],
       'LE CH': [3, 3, 1, 2, 2, 2, 4, 4, 3],
      'LE USA': [28, 28, 26, 27, 24, 24, 22, 21, 19],
    }
columns =['Jahr','Daten','GDP USA','GDP CH', 'LE USA', 'LE CH']
dr = pd.DataFrame(data=d, columns=columns)
```


```python
df0 = pd.read_excel('HE_GDP.xls')
df1 = pd.DataFrame(df0.values[:,2:4], columns=df0.columns[2:4], index=df0.values[:,0])
df1['share'] = df1['Government/Compulsory']/(df1['Government/Compulsory']+df1['Voluntary/Out-of-pocket'])
df2 = df1.sort_values('share')
```


```python
def autolabel_2(rects,bottom):
    #Attach a text label above each bar displaying its height
    for i,rect in enumerate(rects):
        height = rect.get_height() 
        plt.text(rect.get_x() + rect.get_width()/2., bottom[i] + 0.1,
                '%4.1f' % height, ha='center', va='bottom', rotation=90,fontsize=11)

def plot_HEGDP():
    N = len(df0.values[:,0])
    color1 = []; color2=[]
    for x in range(N): 
        color1.append('lightgreen')
        color2.append('orange')
    color1[1]='blue'; color2[1]='red'  
    width = 0.75 
    px = np.arange(N) 
    py1 = df0.values[:,2]
    py2 = df0.values[:,3]

    figX = 18; figY = 9
    fig = plt.figure(figsize=(figX, figY), facecolor='white')
    p0 = plt.bar(px, py1+py2,width, color=color1 )
    p1 = plt.bar(px, py1,    width, color=color1 ,label=df0.columns[2])
    p2 = plt.bar(px, py2,    width, bottom = py1,color=color2,label=df0.columns[3])

    plt.suptitle('Health expenditure as a share of GDP, 2016', fontsize=25, fontweight='bold', color='grey')
    plt.title('OECD, HealthOutlook 2017',fontsize=20)
    plt.ylabel(r'%',  fontsize=40); # plt.xlabel(r'State', fontsize=40)
    plt.legend(loc='upper right',prop={'size': 20})
    ty = np.arange(19)
    plt.xticks(px, df1.index, fontsize=15, rotation=90)
    plt.yticks(ty, fontsize=12)
    autolabel_2(p1,np.zeros_like(py1)); autolabel_2(p2,py1); autolabel_2(p0,py1+py2)                                               
    plt.show()
```


```python
def plot_HEGDP_sort():
    N = len(df0.values[:,0])
    color3 = []; 
    for x in range(N): color3.append('cyan')
    color3[13]='orange'; 
    width = 0.75 
    px = np.arange(N) 
    py1 = df2.values[:,2]

    figX = 18; figY = 6
    fig = plt.figure(figsize=(figX, figY), facecolor='lightgrey')
    p0 = plt.bar(px, py1,width, color=color3)

    plt.suptitle(r'Anteil(%) = $\frac{staatliche Finanzierung}{staatliche Finanzierung + private Finanzierung}$', fontsize=25, fontweight='bold')
    #plt.title('OECD, HealthOutlook 2017',fontsize=20)
    plt.ylabel(r'%',  fontsize=40); # plt.xlabel(r'State', fontsize=40)
    plt.legend(loc='upper right',prop={'size': 20})
    ty = np.linspace(0,1,11)
    plt.xticks(px, df2.index, fontsize=15, rotation=90)
    plt.yticks(ty, fontsize=12)
    autolabel_2(p0,np.zeros_like(py1));                                              
    plt.show()
```


```python
def plot_LE():
    yr = np.array(dr['Daten'])
    LECH = np.array(dr['LE CH'])
    with plt.style.context('fivethirtyeight'): 
        fig = plt.figure(figsize=(10,5))
        axes1 = fig.add_subplot(1, 1, 1)
        axes1.patch.set_facecolor('navajowhite')
        plt.plot(dr['Daten'][::-1], -dr['LE CH'][::-1],  lw=8, label='Schweiz')
        plt.plot(dr['Daten'][::-1], -dr['LE USA'][::-1], lw=8, label='USA')
        plt.fill_between(dr['Daten'][::-1], -dr['LE CH'][::-1], -dr['LE USA'][::-1],facecolor='w', alpha=0.5)
        plt.title('Lebenserwartung USA/Schweiz: Rang in OECD',  fontsize=30) 
        plt.legend(loc=5,prop={'size': 20})
        plt.xlim(1998,2016)
        plt.show()  
```
