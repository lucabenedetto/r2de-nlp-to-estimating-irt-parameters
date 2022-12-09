from setuptools import find_packages, setup

setup(
    name='r2de',
    packages=find_packages(),
    version='0.2.0',
    description='Module for the estimation of difficulty and discrimination of multiple-choice questions from text',
    author='Luca Benedetto',
    license='',
    install_requires=[
        "nltk==3.7",
        "numpy==1.23.5",
        "pandas==1.5.2",
        "pyirt==0.3.4",
        "scikit-learn==1.1.3",
        "matplotlib==3.6.2",
    ],
)
