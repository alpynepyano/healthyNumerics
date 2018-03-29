Title: Loops over the time
Date: 2017-12-05 18:30
Category: devTec
Tags: Python
Slug: python-loops-over-the-time
Cover: /p6004/img6004/curves.jpg
Authors: Peter Schuhmacher
Summary: How to create Python loops in the date-time format


```python
import datetime
import numpy as np
float_formatter = lambda x: "%8.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
```

### Loops over the time

When dealing with solar driven energy balances we need a loop over the time. We can use the Pyton module `datetime` that provide us  with some useful utilities. The loop over the days gives us the *number of day of a year* that we will use to compute the position of the sun.

#### For loop


```python
firstDate = datetime.datetime(2018,2,25,0,0,0)
lastDate  = datetime.datetime(2018,3, 5,0,0,0)
daySteps  = 2

date = firstDate
while date <= lastDate:
    NumberOfDay = date.timetuple()[7]
    print(date, '    number of the day of the year: ',NumberOfDay)
    date += datetime.timedelta(days=daySteps)
```

    2018-02-25 00:00:00     number of the day of the year:  56
    2018-02-27 00:00:00     number of the day of the year:  58
    2018-03-01 00:00:00     number of the day of the year:  60
    2018-03-03 00:00:00     number of the day of the year:  62
    2018-03-05 00:00:00     number of the day of the year:  64
    

To be more flexible we will use a **double loop**. The outer loop determines the days of the year we are interested in, 1 day each month e.g. . The inner loop runs in secondes over the cycle of a day.


```python
def runDay(date,secSteps):    
    startDay = date.replace(hour= 0, minute= 0, second= 0)
    endDay   = date.replace(hour=23, minute=59, second=59)
    day = startDay
    while day <= endDay:
        print('       -->',day)
        day += datetime.timedelta(seconds=secSteps)
            
firstDate = datetime.datetime(2018,2,25,0,0,0)
lastDate  = datetime.datetime(2018,3,5,0,0,0)
daySteps  = 2
secSteps = 4.0 * 3600

date = firstDate
while date <= lastDate:
    NumberOfDay = date.timetuple()[7]
    print(date, '    number of the day of the year: ',NumberOfDay)
    runDay(date,secSteps)
    date += datetime.timedelta(days=daySteps)
```

    2018-02-25 00:00:00     number of the day of the year:  56
           --> 2018-02-25 00:00:00
           --> 2018-02-25 04:00:00
           --> 2018-02-25 08:00:00
           --> 2018-02-25 12:00:00
           --> 2018-02-25 16:00:00
           --> 2018-02-25 20:00:00
    2018-02-27 00:00:00     number of the day of the year:  58
           --> 2018-02-27 00:00:00
           --> 2018-02-27 04:00:00
           --> 2018-02-27 08:00:00
           --> 2018-02-27 12:00:00
           --> 2018-02-27 16:00:00
           --> 2018-02-27 20:00:00
    2018-03-01 00:00:00     number of the day of the year:  60
           --> 2018-03-01 00:00:00
           --> 2018-03-01 04:00:00
           --> 2018-03-01 08:00:00
           --> 2018-03-01 12:00:00
           --> 2018-03-01 16:00:00
           --> 2018-03-01 20:00:00
    2018-03-03 00:00:00     number of the day of the year:  62
           --> 2018-03-03 00:00:00
           --> 2018-03-03 04:00:00
           --> 2018-03-03 08:00:00
           --> 2018-03-03 12:00:00
           --> 2018-03-03 16:00:00
           --> 2018-03-03 20:00:00
    2018-03-05 00:00:00     number of the day of the year:  64
           --> 2018-03-05 00:00:00
           --> 2018-03-05 04:00:00
           --> 2018-03-05 08:00:00
           --> 2018-03-05 12:00:00
           --> 2018-03-05 16:00:00
           --> 2018-03-05 20:00:00
    

#### An iterable time list
If an iterable list is preferred the follwing construct can be used:


```python
d0 = datetime.datetime(2018,3,21, 0, 0, 0)
d1 = datetime.datetime(2018,3,21,23,59,59)
secSteps = 6. * 3600.
dt = datetime.timedelta(seconds = secSteps)
dates = np.arange(d0, d1, dt).astype(datetime.datetime)

for date in dates: 
    print(date)
```

    2018-03-21 00:00:00
    2018-03-21 06:00:00
    2018-03-21 12:00:00
    2018-03-21 18:00:00
    

#### Implicit for loop with iterable list

To build up a time series which is generated as output of a function the implicit for loop over the points of time can be used:

    A = np.array( [solarRadiation(dates[k]) for k in range(0,len(dates))] )
    

#### Solar elevation and iradiation
We give here a procedure to compute the elevation and iraditaion of the sun


```python
def solarRadiation(day):
    Latit = 47.
    rad = np.pi/180.
    NumberOfDay = day.timetuple()[7]                                           # number of days since 1st January
    TMST = day.timetuple()[3]*3600.0 + day.timetuple()[4]* 60.0 + day.timetuple()[5] # True mean solar time in sec
    declin = rad*23.45*np.sin(rad*(280.1 + 0.987*NumberOfDay));                # Declination
    Ht = -np.pi*(43200.0-TMST)/43200.0;                                        # hourley angel                                        
    sinHs =  np.sin(rad*Latit)*np.sin(declin)  \
           + np.cos(rad*Latit)*np.cos(declin)*np.cos(Ht);
    sElevation = np.arctan(sinHs/np.sqrt(1.0-sinHs*sinHs));                    # elevation of sun
    sAzimut = np.cos(declin)*np.sin(Ht)/np.cos(sElevation);                    # azimuth of sun
    kn = NumberOfDay * 2.0*np.pi/365.0;
    I0 = 1353.0 + 45.33*np.cos(kn)   + 0.88* np.cos(2*kn) \
                + 0.005*np.cos(3*kn) + 1.8 * np.sin(kn)   \
                + 0.10 *np.sin(2*kn)+ 0.184* np.sin(3*kn)                       # solar constant
    #S0:= I0 * sinHs                                                            # extraterrestrial radiation
    return I0 * sinHs, Ht
```

#### Daily course of solar radition for some selected months
For March, June, September and December of a year we take day 21 and plot the daily corse of the solar extraterestrial raditaion



```python
firstDate = datetime.datetime(2017, 3, 21, 0, 0, 0)
lastDate  = datetime.datetime(2018, 3, 20, 0, 0, 0)
daySteps  = 1
monthSteps= 3
secSteps  = 3.0 * 3600
dt = datetime.timedelta(seconds = secSteps)

date = firstDate
while date <= lastDate:
    nextDay = date.timetuple()[2] + 1
    
    d0 = date.replace(hour= 0, minute= 0, second= 0)
    d1 = d0.replace(day = nextDay)    
    
    Time= np.arange(d0, d1, dt).astype(datetime.datetime)
    print(Time);print()
    A  = np.array( [solarRadiation(Time[t]) for t in range(0,len(Time))] )
    
    #date += datetime.timedelta(days=daySteps)
    date += relativedelta(months= monthSteps)
```

    [datetime.datetime(2017, 3, 21, 0, 0) datetime.datetime(2017, 3, 21, 3, 0)
     datetime.datetime(2017, 3, 21, 6, 0) datetime.datetime(2017, 3, 21, 9, 0)
     datetime.datetime(2017, 3, 21, 12, 0)
     datetime.datetime(2017, 3, 21, 15, 0)
     datetime.datetime(2017, 3, 21, 18, 0)
     datetime.datetime(2017, 3, 21, 21, 0)]
    
    [datetime.datetime(2017, 6, 21, 0, 0) datetime.datetime(2017, 6, 21, 3, 0)
     datetime.datetime(2017, 6, 21, 6, 0) datetime.datetime(2017, 6, 21, 9, 0)
     datetime.datetime(2017, 6, 21, 12, 0)
     datetime.datetime(2017, 6, 21, 15, 0)
     datetime.datetime(2017, 6, 21, 18, 0)
     datetime.datetime(2017, 6, 21, 21, 0)]
    
    [datetime.datetime(2017, 9, 21, 0, 0) datetime.datetime(2017, 9, 21, 3, 0)
     datetime.datetime(2017, 9, 21, 6, 0) datetime.datetime(2017, 9, 21, 9, 0)
     datetime.datetime(2017, 9, 21, 12, 0)
     datetime.datetime(2017, 9, 21, 15, 0)
     datetime.datetime(2017, 9, 21, 18, 0)
     datetime.datetime(2017, 9, 21, 21, 0)]
    
    [datetime.datetime(2017, 12, 21, 0, 0)
     datetime.datetime(2017, 12, 21, 3, 0)
     datetime.datetime(2017, 12, 21, 6, 0)
     datetime.datetime(2017, 12, 21, 9, 0)
     datetime.datetime(2017, 12, 21, 12, 0)
     datetime.datetime(2017, 12, 21, 15, 0)
     datetime.datetime(2017, 12, 21, 18, 0)
     datetime.datetime(2017, 12, 21, 21, 0)]
    
    

    T = 272. + 15. + 10. * np.roll(I , int((3.*3600.)/secSteps))/max(I)

We run the model for a mid summer day


```python
d0 = datetime.datetime(2018,6,21,0,0,0)
d1 = datetime.datetime(2018,6,22,0,0,0)
secSteps  = 3.0 * 3600
dt = datetime.timedelta(seconds = secSteps)


Time  = np.arange(d0, d1, dt).astype(datetime.datetime)
A  = np.array( [solarRadiation(Time[t]) for t in range(0,len(Time))] )
I  = np.maximum(A[:,0], 0.)   # A[A< 0] = 0; sets negative values to zero
T  = 272. + 15. + 10. * np.roll(I,int((3.*3600.)/secSteps))/max(I)
```


```python
print('Time'); print(Time); 
```

    Time
    [datetime.datetime(2018, 6, 21, 0, 0) datetime.datetime(2018, 6, 21, 3, 0)
     datetime.datetime(2018, 6, 21, 6, 0) datetime.datetime(2018, 6, 21, 9, 0)
     datetime.datetime(2018, 6, 21, 12, 0)
     datetime.datetime(2018, 6, 21, 15, 0)
     datetime.datetime(2018, 6, 21, 18, 0)
     datetime.datetime(2018, 6, 21, 21, 0)]
    


```python
print('A'); print(A); 
```

    A
    [[ -438.24    -3.14]
     [ -198.24    -2.36]
     [  381.15    -1.57]
     [  960.54    -0.79]
     [ 1200.53    -0.00]
     [  960.54     0.79]
     [  381.15     1.57]
     [ -198.24     2.36]]
    


```python
print('I'); print(I);
```

    I
    [    0.00     0.00   381.15   960.54  1200.53   960.54   381.15     0.00]
    


```python
print('T'); print(T);
```

    T
    [  287.00   287.00   287.00   290.17   295.00   297.00   295.00   290.17]
    
