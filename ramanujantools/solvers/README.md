# Ramanujan Solvers

This library contains several "solvers" for both polynomial continued fractions and conservative matrix fields.

## Euler continued fractions

A **generalized continued fraction** is an expansion of the form
$$\mathbb{K}_1^\infty \frac {b_i}{a_i} := \cfrac{b_1}{a_1+ \cfrac{b_2}{a_2+ \cfrac{b_3}{a_3+ \ddots}}}$$

We call it a **polynomial continued fraction** if $a_i=a(i), b_i=b(i)$ where both $a(x), b(x)$ are polynomials (where in most cases, we would also want them to be integral $a(x),b(x)\in\mathbb{Z}[x]$).

In general, it is not easy to find out what is the limit of the continued fraction above. However, there are cases where these can be transformed into "simple" infinite sum expansions.
An **Euler polynomial continued fraction** is a continued fraction as above where

```math
\begin{align*}
b\left(x\right) & =-h_{1}\left(x\right)h_{2}\left(x\right)\\
f\left(x\right)a\left(x\right) & =f\left(x-1\right)h_{1}\left(x\right)+f\left(x+1\right)h_{2}\left(x+1\right)
\end{align*}
```

for polynomials $h_1(x), h_2(x), f(x)$.

If this holds, then we have

```math
\mathbb{K}_1^\infty \frac {b(i)}{a(i)}= \frac{f(1)h_2(1)}{f(0)}  \left(\left( \sum_{k=0}^\infty \frac{f(0)f(1)}{f(k)f(k+1)} \prod_{i=1}^{k} \frac{h_1(i)}{h_2(i+1)} \right)^{-1} - 1\right)
```

To find if a polynomial continued fraction is in the Euler family, use `ramanujantools.pcf.euler_family.EulerSolver' and call

> `EulerSolver.solve_for(a,b)`

See the `euler_pcf_test.py` file for more examples, or go to the [python notebook here](https://colab.research.google.com/drive/10aJ22X9LMhP_NNJCrcpDe0YLxXmTEfz3?usp=sharing).

## Coboundary Equivalence

Given two polynomial matrices $M_1(x), M_2(x)$, we say that they are **$U(x)$-coboundary equivalent** for a polynomial matrix $U(x)$ if

```math
M_1(x) \cdot U(x+1) = U(x) \cdot M_2(x).
```

The library `ramanujantools.cmf.coboundary` contains the `CoboundarySolver` class which can be used to
find such $U(x)$ solutions.

In general, the matrices $M_1, M_2$ can also depend on other variables, so we can look for a coboundary equivalence
of a family of pairs of polynomial matrices.
Finally, given a single polynomial matrix $M(x,y)$, we can take $M_1 = M(x,y)$ and $M_2=M(x,y+1)$. If they are
coboundary equivalent with some matrix $U(x,y)$, then this means that the pair
$M(x,y), U(x,y)$ form a conservative matrix field.

To see examples how to use this solver, see the `coboundary_test.py` file, and also you can
check the [python notebook here](https://colab.research.google.com/drive/1SO0KPax6dYo7OD27I5TECCBx5uWIZXDi?usp=sharing)

## FFbar Solver

The FFbar is a CMF construction involving two functions $f(x, y), \bar{f}(x, y)$.

If $f(x, y), \bar{f}(x, y)$ satisfy the following conditions:
- linear condition: $f(x+1, y-1) - \bar{f}(x, y-1) + \bar{f}(x+1, y) - f(x, y) = 0$
- quadratic condition: $f\bar{f}(x, y) - f\bar{f}(x, 0) -f\bar{f}(0, y) + f\bar{f}(0, 0) = 0$, where $f\bar{f}(x, y) = f(x, y) \cdot \bar{f}(x, y)$

Then the following matrices form a CMF:
    $$Mx = \begin{pmatrix} 0, b(x) \cr 1, a(x, y) \end{pmatrix}$$
    $$My = \begin{pmatrix} \bar{f}(x, y), b(x) \cr 1, f(x, y) \end{pmatrix}$$

Where $a(x, y) = f(x, y) - \bar{f}(x+1, y) = f(x+1, y-1) - \bar{f}(x, y-1)$, and $b(x) = f\bar{f}(x, 0) - f\bar{f}(0, 0) = f\bar{f}(x, y) - f\bar{f}(0, y)$,

The `FFBarSolver` supports two functionalities:

```python
FFbarSolver.solve_ffbar(f: sp.Expr, fbar: sp.Expr) -> list[FFbar]
```
Given two functions with symbolic parameters $f(x, y), \bar{f}(x, y)$, it returns a list of FFbar for all parameterizations that satisfy the conditions above.

```python
FFbarSolver.from_pcf(pcf: PCF) -> list[FFbar]
```
Given a PCF, attempts to find FFbar constructions such that the M(x) matrix of the FFbar is the matrix representation of the PCF.
