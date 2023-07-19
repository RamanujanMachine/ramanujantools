from setuptools import setup

setup(
    name="ramanujan",
    version="0.0.1",
    python_requires=">=3.8.10",
    description="The official research tools of Ramanujan group",
    packages=[
        "ramanujan",
        "ramanujan.pcf",
        "ramanujan.cmf",
        "ramanujan.cmf.ffbar",
        "ramanujan.cmf.known_cmfs",
    ],
    install_requires=[
        "sympy>=1.11.1",
        "scipy>=1.10.1",
        "gmpy2>=2.1.5",
    ],
)
