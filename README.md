# Binomial Tree for Options Pricing
### Motivation:
Started as a project for learning in relation to schoolm but turned out as a tool that 
could be useful for more.

## What does it do?
This class will construct/calculate/make/whatever a binomial tree with given parameters 
and calculate option prices for both American and European Call and Put options.

- The class will make tree objects with Dataframe representation 
(except for non-recombining trees).
- The `write()` method will generate an Excel file with the result.

### Dividends
The class can take dividends into account (dividend yield **and** discrete dividends). 
It has 2 ways of solving for discrete dividends, the 'F solution' and 'non-recombining'.

#### F solution
Subtracts the present value from current spot, and makes binomial tree based on this
'pre-paid forward'. 

#### Non-recombining tree
As the name suggests, the tree does not recombine after dividend payout and calculates
the option premium as normal (assumes that we can trade just before and after 
stock goes ex-div)

***

# Full parameter specification
[Parameters specification](docs/parameters.md)

***

# How to pass parameters/arguments:
There are 2 ways of parsing arguments:
1. A dictionary
2. Keyword arguments (kwargs)
   
If both a dictionary and keyword arguments are parsed they will join together. 
Keyword arguments will override any parameters passed in both the dictionary and 
as keyword arguments.

***

# Which parameters to pass:
*If nothing is passed, the `help()` method will print help for specifying parameters*  


## Spot and strike
#### Both must be passed
- `spot`
- `strike`

***

## Time and period specification
Must be in terms of years (e.g., maturity in 6 months would be *T = 6/12* ).  
The variable `dtfreq` can be passed as a string (`'d'`, `'w'`, or `'m'`) for
prettier formatting in output.
#### 2 of 3 must be passed   
- `T`
- `dt`
- `periods`

***

## Interest rate, dividend yield, and continuous compounding
#### 0 of 3 *must* be passed
- `r`
- `rcont`
- `divyield`

***

## Volatility and up/down movements
#### 1 of 3 must be passed
- `vola`
- `u`
- `d`

For custom function for calculating up/down movements, 
see [udfunc specification](docs/parameters.md#udfunc).

***

## Dividends
#### 0 of 2 *must* be passed
- `discdiv`
- `nonrec`

***

## Directory for Excel output
#### 0 of 4 *must* be passed
- `direc`
- `folname`
- `fname` 
- `write`

***

## Various
#### 0 of 5 *must* be passed
- `collapsed`
- `maketrees`
- `headerformat`
- `rounding`
- `makedfs`

***

# Examples of use
### imports
```python
from main import binomialTrees
```

***

## Normal tree
Simple 3-period binomial tree:

```python
parsNormal = dict(fname = 'NormalTree', 
                  spot = 100, strike = 95, 
                  dt = 1/12, periods = 3, 
                  vola = 0.20, r = 0.05, 
                  dtfreq = 'm', rounding = 4)  

binoNormal = binomialTrees(params = parsNormal, showIntrinsic = True, periods = 4)
```

***

## F tree
For discrete dividends, the F solution will be the default as it is faster than nonrec:

```python
parsF = dict(fname = 'Fsol',
             spot = 41, strike = 40,
             dt = 4/12, periods = 3,
             vola = 0.20, r = 0.08,
             showIntrinsic = True, write = True, dtfreq = 'm',
             discdiv = [(4/12, 5), (8/12, 5)])

binoF = binomialTrees(params = parsF)
```

***

## Non-recombining tree
For discrete dividends -> non-recombining tree.
Can specify which options to make through the `maketrees` parameter.

```python
pars = dict(direc = '/Users/EspenHamre/Desktop',
            fname = 'NonRec',
            spot = 100, strike = 95,
            vola=0.20, r = 0.03,
            dt = 1/12, periods = 3,
            dtfreq = 'm', headerformat = 'dt', showIntrinsic = False,
            discdiv = [(1/12, 2)],
            nonrec = True, makedfs = False, maketrees = ['ec', 'ep'])

binoNonrec = binomialTrees(params = pars, test = True)
```

***

# Object as callable
The binomialTrees object is callable, meaning an instance of the object can be 
used as a function.  
This can be handy for getting values in loops or quickly getting new values without 
re-specifying all the parameters.  
As a default the class will make american call and put, aswell as european call and put, 
unless the `maketrees` keyword states otherwise. If only one or some specific option types
are needed, it would be more efficient to specify which types to make.

### Example:
*As periods becomes large (i.e., dt becomes small), the rounding parameter needs to be 
sufficiently large to avoid miscalculation.*
```python
parsNormal = dict(fname = 'NormalTree', 
                      spot = 100, strike = 95, 
                      dt = 1/12, periods = 3, 
                      vola = 0.20, r = 0.05, 
                      dtfreq = 'm', rounding = 8)  

binoNormal = binomialTrees(params = parsNormal, maketrees = ['ec', 'ep'])

ec_ep_50periods = binoNormal(['ecOptionPrice', 'epOptionPrice'], periods = 50)  
```

    >> ec_ep_50periods  
    {'ecOptionPrice': 7.71749875, 'epOptionPrice': 1.5373898}

Returns the european call and put price with all parameters remaining the same, 
except periods being 50.  


### In a loop:
```python
ecList = []

for i in range(1, 100):
    ecList.append(binoNormal(['ecOptionPrice'], maketrees = ['ec'], periods = i))
```
Makes a list of the european call price.



***