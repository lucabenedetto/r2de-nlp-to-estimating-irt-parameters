from setuptools import find_packages, setup

setup(
    name='r2de',
    packages=find_packages(),
    version='0.1.0',
    description='Module for the estimation of difficulty and discrimination of multiple-choice questions from text',
    author='Luca Benedetto',
    license='',
    install_requires=[
        "matplotlib==3.1.2",
        "nltk==3.4.5",
        "numpy==1.22.0",
        "pandas==0.25.3",
        "pyirt==0.3.4",
        "scikit-learn==0.22.1",
    ],
)
