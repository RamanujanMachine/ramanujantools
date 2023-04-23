from setuptools import setup

setup(
    name="cmf",
    version="0.0.1",
    python_requires=">=3.8.10",
    description="Research tools for Conservative Matrix Fields",
    packages=["cmf", "cmf.ffbar", "cmf.known_cmfs"],
    install_requires=[
        "sympy>=1.11.1",
        "scipy>=1.10.1",
    ],
)
