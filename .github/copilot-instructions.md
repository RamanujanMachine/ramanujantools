# Copilot Instructions — ramanujantools

You are an AI developer working on `ramanujantools`, a Python library for
symbolic and numeric matrix operations used by the
[Ramanujan Machine](https://www.ramanujanmachine.com/) project.

## Project overview

`ramanujantools` provides tools for Conservative Matrix Fields (CMFs),
Polynomial Continued Fractions (PCFs), linear recurrences, and high-precision
constant computation. The numeric backend uses
[python-flint](https://github.com/flintlib/python-flint) for exact rational
arithmetic.

## Architecture

- `ramanujantools/matrix.py` — `Matrix` (SymPy-based), public API entry point.
- `ramanujantools/flint_core/numeric_matrix.py` — `NumericMatrix` (FLINT `fmpq_mat` wrapper).
- `ramanujantools/flint_core/symbolic_matrix.py` — `SymbolicMatrix`.
- `Matrix.walk()` dispatches to `NumericMatrix.walk()` or `SymbolicMatrix.walk()` based on input type.
- `@batched` decorator (`ramanujantools/utils/batched.py`) allows `iterations` to be scalar or list.
- `NumericMatrix.lambda_from_rt()` compiles a SymPy matrix into a fast FLINT evaluator.
- `Matrix._walk_inner` has an LRU cache; clear with `Matrix._walk_inner.cache_clear()` before benchmarks.

## Code conventions

### Testing
- Tests live next to source as `*_test.py` files (e.g., `numeric_matrix_test.py`).
- Run with `pytest`. Benchmarks use `pytest-benchmark`.
- Benchmarks live in `matrix_benchmark.py` and run **strictly via pytest**, never as standalone scripts.
- Every new public function needs at least one test (happy path + edge case).
- Tests should exercise the public `Matrix` API, not internal classes directly, unless testing internal-only behavior.

### Numeric precision
- **Never use Python `float`** for mathematical verification. Use `mpmath.mpf`, `sympy.Rational`, or FLINT types.
- Set `mpmath.mp.dps` to at least 2× the digits you need.
- Verify formulas to 100+ decimal places.

### Style
- Clear, descriptive variable names (e.g., `first`/`last` not `lo`/`hi`).
- Extract magic numbers into named constants (e.g., `SEQUENTIAL_THRESHOLD = 8`).
- Docstrings on public functions: one-line summary, parameters, return value.
- No unnecessary abstractions for one-off operations.

### Performance
- Profile before optimizing (`cProfile`, `timeit`).
- Benchmark before and after, at realistic scale (N ≥ 1000 for walks).
- Binary splitting (`_product_tree`) is used in `NumericMatrix.walk()` — balanced divide-and-conquer is critical for FLINT rationals where entry sizes grow with each multiplication.

## PR review workflow

When responding to PR review comments:
1. Read every review comment carefully before making changes.
2. Prefer the simplest fix that addresses the reviewer's concern.
3. Run the relevant tests after every change (`pytest <test_file> -v`).
4. If the reviewer asks to check for existing utilities/packages, do the research and report findings even if no suitable alternative exists.
5. Commit with a clear message referencing which review comments are addressed.

## Reviewer preferences (RotemKalisch)

- Benchmarks must be pytest-only — no `if __name__ == "__main__"` standalone scripts.
- Test through the public `Matrix` class API where possible.
- Check for existing pythonic utilities/packages before writing custom algorithms.
- Justify any new dependency with a comparison of how the code looks and performs with vs. without it.

## Repository rules

- Do not push to `master` directly. Use feature branches and PRs.
- Run the full test suite (`pytest`) before pushing.
- The CI runs `python-package.yml` — ensure it passes.
