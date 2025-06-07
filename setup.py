from setuptools import setup, find_packages

setup(
    name="MCPower",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas", 
        "matplotlib",
        "scipy",
        "scikit-learn"
    ],
    author="Paweł Lenartowicz",
    description="Monte Carlo Power Analysis for Linear Models",
)
