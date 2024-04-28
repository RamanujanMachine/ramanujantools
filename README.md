# Ramanujan Tools

The official and up-to-date research tools of the Ramanujan Machine research group.

## Installation

```commandline
pip install git+https://github.com/RamanujanMachine/ramanujantools.git
```

## The code

### Overview

The most relevant pieces of code of this library are:

- `Matrix` which inherits `sympy.Matrix` and adds the walk method,
  which allows us to walk alongside a trajectory
- `PCF` (Polynomial Continued Fraction) which can calculate the limit of a PCF
- `CMF` (Conservative Matrix Field) which is made of instances of `Matrix`
  and methods such as walk and limit
- `FFbar` which contains the conditions and logic for ffbar construction of CMFs
- The `known_cmfs` module which contains most of our known CMFs
- CMF to PCF transformation functions

### Example: calculating the limit of zeta3 CMF alongside the diagonal

```python
from ramanujantoolstools.cmf import known_cmfs

known_cmfs.zeta3().limit([1,1], 100).as_float()
```
