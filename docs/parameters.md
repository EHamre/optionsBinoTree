# Parameter specification

How each parameter should be parsed.  

***

## Directories for excel output

### `direc`
Directory for Excel output as string.  
e.g., `'Users/username/Desktop'`  
Default: `None` → will select working directory.

### `folname`
Optional folder for output in chosen directory (could be useful in combination with 
callable functionality).  
e.g., `'myFolderName'`  
Default: `None`

### `fname`
filename as string (not including filetype suffix) e.g., `'myFilename'`  
Default: `'binotree'`  

***

## Spot and strike

### `spot`  
Numerical value for current spot, e.g. `100`  


### `strike`  
Numerical value for strike on options, e.g. `95`  

***

## Time and periods
T, dt -> periods = T/dt  
T, periods -> dt = T/periods  
dt, periods -> T = dt * periods  
T, dt, periods -> dt = T/periods  

### `T`  
Time to maturity in terms of years, e.g.,   
- 30 days: `30/365`  
- 3 weeks: `3/52`  
- 3 months: `3/12`  
- 2 years: `2`  

### `dt`  
Length of each period in terms of years, e.g.,  
- 1 day: `1/365`  
- 1 week: `1/52`  
- 1 month: `1/12`  
- 1 year: `1`  

### `periods`  
Number (integer) of periods in binomial tree, e.g. 3  

### `dtfreq`  
Needed for formatting.   
Either of: 
- `'d'`
- `'w'`
- `'m'`

***

## Interest rate, dividend yield, and continuous compounding
### `r`
Yearly risk-free interest rate in decimal percentage,  
e.g., For p.a. 4%: `0.04`  

### `rcont`
Boolean to determine continous or discrete compunding interest rate and dividend yield,  
e.g.,  
Continous: `True`  
Discrete: `False`

### `divyield`
Yearly dividend yield in decimal percentage,  
e.g., for p.a. 2% dividend yield: `0.04`

***

## Volatility and up/down movements
See `help("updown")` method for priority specification 
(for example what happens when `u` and `vola` are parsed)

### `vola`
Volatility in terms of yearly standard deviation (sigma), e.g., `0.20`

### `u`
Up factor for each node movement up, e.g., `1.10`

### `d`
Down factor for each node movement down, e.g., `0.90`

### Default up/down calculation  
<img src="https://raw.githubusercontent.com/EHamre/optionsBinoTree/main/docs/images/uFormula.png" width="40%" height="40%">
<img src="https://raw.githubusercontent.com/EHamre/optionsBinoTree/main/docs/images/dFormula.png" width="40%" height="40%">


### udfunc  
`udfunc` can be passed to specify a custom function for calculating up/down movements.
Function must have `**kwargs`.  
All parameters must/should be specified as keyword arguments to avoid complications.

#### Can take any of these parameters:  
- vola 
- T 
- dt 
- periods 
- r 
- divyield 
- discountRate
- discountDiv 
- discountRateMinusDiv 
- spot 
- strike  

#### Example
```python
def udfuncNEW(r, divyield, dt, vola, **kwargs):
    import numpy as np
    
    u = np.exp((r-divyield)*dt + vola*np.sqrt(dt))
    d = np.exp((r-divyield)*dt - vola*np.sqrt(dt))
    
    return u, d

bino_Custom_udfunc = bt.tree(params = pars, udfunc = udfuncNEW)
```

***

## Discrete dividends
### `discdiv`
List with lists/tuples with time for dividend first followed by dividend payout e.g.,  
```python
discdiv = [(1/12, 2), (4/12, 5)]
```  


### `nonrec`
Boolean which determines if tree is non-recombining, if False → F-solution.  
Default: `False`

***

## Formatting & Writing
### `collapsed`
Boolean which determines if tree is collapsed.  
If `True` → no empty cells between nodes.  
Default: `False`

### `showIntrinsic`
Boolean which determines whether to display instrinsic value or not.  
Default: `True`

### `portfolios`
Boolean which determines if replicating portfolios are added
excel output.  
If `True` → no empty cells between nodes.  
Default: `False`

### `write`
Boolean which determines if excel output written directly from construction.  
Default: `False`

### `headerformat`
String which determines if header is formated in terms of periods or
'actual' time:  
`'periods'` or `'dt'`  

### `rounding`
Integer specifying rounding for decimals, e.g., `rounding = 4`  
Default: `2`

### `portfolios`
Boolean specifying whether replicating portfolios are to be written,  
`True` or `False`  
Default: `False`

***

## maketrees and makedfs
### `maketrees`
List of options to calculate e.g., for calculating european call and american put  
```python
maketrees = ['ec', 'ap']
```
Default: `maketrees = ['ec', 'ep', 'ac', 'ap']`

### `makedfs`
Boolean determining if dataframes are to be constructed (speeds up code when False).  
`True` or `False`
Default: `True`

***

# Time specification in callable functionality
Which time parameters to discard/recalculate when object is called:
- new T       
  - keep dt → new periods
- new dt      
  - keep T → new periods
- new periods 
  - keep T → new dt
    

- 2 or 3 new  
  - act as normal construction (`__init__`)

***

**Make Dataframes:**  
If Dataframes are to be made from `__call__`, you must parse 'dfs' in toreturn parameter.



***