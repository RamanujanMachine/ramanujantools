This library contains Conservative Matrix Field (CMF) code and research tools.

## Installation

```commandline
pip install git+https://github.com/RamanujanMachine/CMF.git
```

## The code
### Overview
The most relevant pieces of code of this library are:
* `Matrix` which inherits `sympy.Matrix` and adds the walk method which allows us to walk alongside a trajectory
* `CMF` which contains two instances of `Matrix` (Mx and My) and methods such as walk and limit
* The `ffbar` module which contains the conditions and logic for ffbar construction of CMFs
* The `known_cmfs` module which contains most of our known CMFs (both general and ffbar-constructed)

### Example: calculating the limit of zeta3 CMF alongside the diagonal
```python
from cmf import CMF
import cmf.known_cmfs  # this might take a few seconds

cmf.known_cmfs.zeta3.limit([1,1], 100)
```
