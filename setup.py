from setuptools import setup

setup(
    name="ramanujantools",
    version="0.0.1",
    python_requires=">=3.10",
    description="The official research tools of Ramanujan group",
    packages=[
        "ramanujantools",
        "ramanujantools.pcf",
        "ramanujantools.cmf",
        "ramanujantools.cmf.ffbar",
        "ramanujantools.cmf.known_cmfs",
        "ramanujantools.flint_core",
    ],
    install_requires=[
        "mpmath>=1.3.0",
        "multimethod>=1.10",
        "pytest>=8.2.0",
        "sympy>=1.11.1",
        "gmpy2>=2.1.5",
        "tqdm>=4.65.0",
        "python-flint>=0.7.0a5",
    ],
)
