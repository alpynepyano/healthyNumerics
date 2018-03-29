Title: Computational Fluid Dynamics 01: Pressure correction as mass consistent flow model
Date: 2017-09-02 08:30
Category: ComputationalFluidDynamics
Tags: flow model, Lagrange multiplier, pressure corection
Slug: cfd_01_mass_consistent_model_pressure_correction
Cover: /p5001/img5001/a_aletschgletscher_B.jpg
Authors: Peter Schuhmacher
Summary: The pressure correction method is presented as Lagrange multiplier method to satisfy mass conservation

## Mathematische Herleitung eines massen-konsistenten Strömungsmodelles

### Fragestellung

Bei der numerischen Lösung der **Navier-Stockes-Gleichung** wird im einzelnen Integrationschhritt das Flussfeld nicht divergenzfrei. Wird das nicht korrigiert, würde durch den numerischen Vorgang künstlich Masse entfernt oder zugeführt, was die Ergbenisse verfälscht. 

Eine ähnliche Fragsellung tritt bei der räumlichen Interpolation von **meteorologischen Windfeld-Daten auf**. Durch eine geomterische/trigonometrische Interpolationsformel wird das berechnete Windfeld nicht divergenzfrei. Hüglige Topographie beispielsweise, die von der Interpolation "nicht gesehen wird", beeinflusst den Massenfluss.

In beiden Fällen liegt ein geschätzes Wind- oder Flussfeld vor, das wir mit $\mathbf{u_o}$ bezeichnen. Es soll ein neues Flussfeld $\mathbf{u}$ gefunden werden. Die Änderung gegenüber $\mathbf{u_o}$ soll klein sein, und sie soll unter der Randbedingung gefunden werden, dass $\mathbf{u}$ divergenzfrei ist (d.h. die Masse bleibt konstant).

Die Fragestellung kann als **Lagrange'sche Optimierungsaufgabe** formuliert werden, wobei ** $\lambda$ ** der Lagrange-Multplikator ist.

Es ist zu beachten, dass das Windfeld/die Flussgrösse $\mathbf{u}$  **3-dimensional** ist, d.h. es hat drei Komponenten in x-, y- und z-Richtung, die oft mit $\mathbf{u} = (u,v,w)$ notiert werden. Ferner ist zu beachten, dass $\mathbf{u}$ mit seinen drei Komponenten an hunderten bis tausenden von Gitternpunkten vorliegen kann. Es handelt sich hier um eine Optimierungsaufgabe, die numerisch gelöst werden muss.


### Die Lagrange Formulierung

$$
\begin{equation}
\begin{array}{rcccl}
&  &  \textrm{Die Änderung}    &  & \textrm {unter der RB, dass}  \\
&  &  \textrm{soll klein sein} &  & \textrm {$\mathbf{u}$ divergenzfrei ist}  \\
L& = &  \iiint \limits_{\Delta V}  \mathbf{A}(\mathbf{u}-\mathbf{u}_o )^2 \, \mathrm{d}V & - & \iiint \limits_{\Delta V} \lambda \cdot \nabla \mathbf{u}  \cdot \mathrm{d}V
\end{array}
\end{equation}
$$

Wenn ein $\mathbf{u}$ gefunden wird, das $L$ minimiert, dann gilt
$$
L(\mathbf{u}) = L_{min} = \iiint \limits_{\Delta V}  \mathbf{A}(\mathbf{u}-\mathbf{u}_o )^2
$$


### Herleitung der Lösung

Zur Bestimmung von $\mathbf{u}$ wählen wir einen __Variationsansatz__. Wir nehmen an, dass wir eine Näherungslösung $(\mathbf{u}+ \delta \mathbf{u} )$ haben, die nahe bei $ \mathbf{u}$ liegt, d.h. $\delta \mathbf{u}$ sei klein. Mit der Berücksichtigung von $\delta \mathbf{u}$ wird das Funktional $L$ zu:

$$\begin{array}{lll}
L(\mathbf{u} + \delta \mathbf{u}) & = & L_{min} + \delta L \\
                                  & = & L(\mathbf{u}) + \delta L(\mathbf{u}) \\
\end{array}
$$

Ziel ist es, eine Lösung $\delta L(\mathbf{u}) = 0$ finden. Dazu setzen wir $L(\mathbf{u} + \delta \mathbf{u})$ in Gleichung (1) ein:

$$\begin{array}{rll}
L(\mathbf{u} + \delta \mathbf{u}) & = & L_1 + L_2  \\
L1& = &  \iiint \limits_{\Delta V}  \mathbf{A}((\mathbf{u} + \delta \mathbf{u})-\mathbf{u}_o )^2 \, \mathrm{d}V  \\
  & = &  \iiint \limits_{\Delta V}  \mathbf{A}((\mathbf{u} - \mathbf{u}_o )+ \delta \mathbf{u})^2 \, \mathrm{d}V \\
  & = & \iiint \limits_{\Delta V}  (\mathbf{A} (\mathbf{u} - \mathbf{u}_o)^2 + 2 \mathbf{A} \delta \mathbf{u} (\mathbf{u} - \mathbf{u}_o ) + \mathbf{A}\delta\mathbf{u}^2 )\, \mathrm{d}V \\
L2& = &  - \iiint \limits_{\Delta V} \lambda \cdot \nabla (\mathbf{u} + \delta \mathbf{u})  \cdot \mathrm{d}V \\
  & = &  - \iiint \limits_{\Delta V} \lambda \cdot \nabla \delta \mathbf{u}  \cdot \mathrm{d}V \\
  & = &  \iiint \limits_{\Delta V} \nabla \lambda \cdot \delta \mathbf{u}   \, \mathrm{d}V  -   \iint \limits_{\Delta \Omega} \lambda \cdot \delta \mathbf{u} \cdot \mathbf{n}\cdot \mathrm{d}S  \\
\end{array}
$$

In L1 werden wir den Term 2. Ordnung $\mathbf{A}\delta\mathbf{u}^2$ vernachlässigen. In L2 setzten wir die getroffene Voraussetzung ein, dass die Divergenz des gesuchten Flussfeldes gleich null ist ($\mathbf{\nabla u = 0}$).
Danach kam das Divergenz-Theorem zur Anwendung. Fassen wir die Terme gemäss dem Variationsansatz zusammen, so erhalten wir:

$$\begin{array}{rll}
L(\mathbf{u} + \delta \mathbf{u}) & = & L_{min} + \delta L \\
                                  & = & L(\mathbf{u}) + \delta L(\mathbf{u}) \\
L(\mathbf{u})                     & = &  \iiint \limits_{\Delta V}  (\mathbf{A} (\mathbf{u} - \mathbf{u}_o)^2  )\, \mathrm{d}V\\
\delta L(\mathbf{u}) & = &\iiint \limits_{\Delta V}  2 \mathbf{A} \delta \mathbf{u} (\mathbf{u} - \mathbf{u}_o ) + 
 \nabla \lambda \cdot \delta \mathbf{u}   \, \mathrm{d}V  -   \iint \limits_{\Delta \Omega} \lambda \cdot  \delta \mathbf{u} \cdot \mathbf{n}\cdot \mathrm{d}S  \begin{equation} \end{equation} \\
\end{array}
$$

Um eine Lösung $\delta L(\mathbf{u}) = 0$ finden, müssen beide Terme in Gleichung (2) gleich null sein. Für den ersten Term verwenden wir wieder die getroffene Voraussetzung, dass die Divergenz des gesuchten Flussfeldes gleich null ist ($\mathbf{\nabla u = 0}$). Um das praktisch ausnutzen zu können, wird vom ganzen Term der Gradient verwendet, was die Bestimmungsgleichung zur Berechnung des Flussfeldes ergibt. Der zweite Term stellt den Rand dar. Hier gibt es zwei Möglichkeiten, um die Null-Lösung zu erreichen. Entweder wird das Flussfeld am Rand nicht verändert, und dann ist $\mathbf{\delta u = 0}$, oder $\lambda = 0$, was eine Flussänderung am Rand ergibt.

### Die numerisch zu lösenden Gleichungen

Der Gradient des ersten Termes von $\delta L(\mathbf{u})$ von (2) ergibt folgende **Bestimmungsgleichung** für das Flussfeld

$$
\begin{equation}
 \nabla^2 \lambda = 2 \mathbf{A}  \nabla\mathbf{u}_o 
\end{equation}
$$

Die **Gleichung für das up-date von u** ergibt sich aus dem ersten Term von $\delta L(\mathbf{u})$ von (2)

$$
\begin{equation}
\mathbf{u} = \mathbf{u_o} -\frac{1}{2A}\nabla\lambda 
\end{equation}
$$


Die **Randbedingung für $\lambda$ ** kann über die up-date-Funktion hergeleitet werden. Sie bestimmt, wie $\mathbf{u_o}$ an den Rändern festgelegt wird:

\begin{equation}
\begin{array}{llll}
\nabla \lambda = 0 & \rightarrow & \mathbf{u} = \mathbf{u_o} &\textrm{keine Änderung von }\mathbf{u_o}\\
\nabla \lambda = value & \rightarrow & \mathbf{u} = \mathbf{u_{value}} &\textrm{Änderung auf vorgegebenen Wert }\mathbf{u_{value}}\\
\lambda = 0 & \rightarrow & \nabla \lambda \neq 0 &\textrm{offener Rand mit Änderung von }\mathbf{u_o}\\
\end{array}
\end{equation}



### Zur algorithmischen Umsetzung

Zur Herleitung haben wir eine koordinaten-freie Darstellung verwendet. Für die Erstellung eines lauffähigen Computer-Programmes muss indessen ein Koordinatensystem gewählt werden. Wir geben hier die Notierung in kartesischen Koordinaten. Es ist allerdings darauf hin zu weisen, dass kaum ein Problem von Relevanz in kartesischen Koordinaten gelöst werden kann, denn das Gitter muss auf geeignete Weise den Oberflächenformen angepasst werden. Der Ausweg dazu ist, dass entweder gekrümmte Koordinaten verwendet werden, oder dass unstrukturierte Gitter verwendet werden, die nicht mehr der logischen Struktur eines strukturierten, rechtwinkligen Gitters folgen.


$$
\nabla^2 \lambda  =    \nabla\mathbf{u}_o
$$

$$
\frac{\partial^2 \lambda}{\partial x^2} + \frac{\partial^2 \lambda}{\partial y^2} + \frac{\partial^2 \lambda}{\partial z^2} = \frac{\partial u_o}{\partial x} +  \frac{\partial v_o}{\partial y} +  \frac{\partial w_o}{\partial z}  
$$


$$
\mathbf{u} = \mathbf{u_o} - \nabla\lambda 
$$

$$
u = u^o -\frac{\partial \lambda}{\partial x}
$$

$$
v = v^o -\frac{\partial \lambda}{\partial y}
$$

$$
w = w^o -\frac{\partial \lambda}{\partial z}
$$

$$
\frac{\partial^2 \lambda}{\partial x^2} = (\lambda_{i+1,j,k} - 2\lambda_{i,j,k} + \lambda_{i-1,j,k} )\;/\; (dx^2)
$$

$$
\frac{\partial u_o}{\partial x} = (u^o_{i+1,j,k} - u^o_{i-1,j,k} ) \;/ \;(2 dx)
$$

### Anhang

$$\begin{array}{rllll}
\iiint \limits_{\Delta V} \nabla (\lambda \cdot \delta u ) \, \mathrm{d}V & = & \iiint \limits_{\Delta V}( \nabla \lambda \cdot \delta u  +  \lambda \cdot \nabla\delta u ) \, \mathrm{d}V &  &\textrm{Produktregel}\\
\iiint \limits_{\Delta V} \nabla (\lambda \cdot \delta u ) \, \mathrm{d}V & = & \iint \limits_{\Delta \Omega} \lambda \cdot \delta u \cdot \mathbf{n}\cdot \mathrm{d}S &  & \textrm{Divergenz-Theorem} \\
\iint \limits_{\Delta \Omega} \lambda \cdot \delta u \cdot \mathbf{n}\cdot \mathrm{d}S & = & \iiint \limits_{\Delta V} \nabla \lambda \cdot \delta u    \, \mathrm{d}V  + \iiint \limits_{\Delta V}\lambda \cdot \nabla\delta u \, \mathrm{d}V &  & \textrm{partielle Integration}\\
-\iiint \limits_{\Delta V}\lambda \cdot \nabla (\mathbf{u} + \delta \mathbf{u}) \, \mathrm{d}V & = & \iiint \limits_{\Delta V} \nabla \lambda \cdot (\mathbf{u} + \delta \mathbf{u})   \, \mathrm{d}V  -   \iint \limits_{\Delta \Omega} \lambda \cdot (\mathbf{u} + \delta \mathbf{u}) \cdot \mathbf{n}\cdot \mathrm{d}S  & & \textrm{partielle Integration}\\
\end{array}$$

###### Postprocessing for the numbering of the equations


```python
%%javascript
MathJax.Hub.Config({TeX: { equationNumbers: { autoNumber: "AMS" } } });
```


    <IPython.core.display.Javascript object>



```python
%%javascript
MathJax.Hub.Queue(["resetEquationNumbers", MathJax.InputJax.TeX],["PreProcess", MathJax.Hub],["Reprocess", MathJax.Hub]);
```


    <IPython.core.display.Javascript object>

