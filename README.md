# ML-DL-implementation
[![Build Status](https://travis-ci.org/RoboticsClubIITJ/ML-DL-implementation.svg?branch=master)](https://travis-ci.org/RoboticsClubIITJ/ML-DL-implementation)
[![Gitter](https://badges.gitter.im/ML-DL-implementation/community.svg)](https://gitter.im/ML-DL-implementation/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/RoboticsClubIITJ/ML-DL-implementation)

Machine Learning and Deep Learning library in python using numpy and matplotlib.

## Why this repository?
-----------------------

This repository gives beginners and newcomers in
the field of AI and ML a chance to understand the
inner workings of popular learning algorithms by presenting them with a simple to analyze the implementation of ML and DL algorithms in pure python using only numpy as a backend for linear algebraic computations for the sake of efficiency.

The goal of this repository is not to create the most efficient implementation but the most transparent one, so that anyone with little knowledge of the field can contribute and learn.

Installation
------------

You can install the library by running the following command,

```python
python3 setup.py install
```

For development purposes, you can use the option `develop` as shown below,

```python
python3 setup.py develop
```
   
Testing
-------

For testing your patch locally follow the steps given below,

1. Install [pytest-cov](https://pypi.org/project/pytest-cov/). Skip this step if you are already having the package.
2. Run, `python3 -m pytest --doctest-modules --cov=./ --cov-report=html`. Look for, `htmlcov/index.html` and open it in your browser, which will show the coverage report. Try to ensure that the coverage is not decreasing by more than 1% for your patch.


## Contributing to the repository

Follow the following steps to get started with contributing to the repository.

- Clone the project to you local environment.
  Use
  `git clone https://github.com/RoboticsClubIITJ/ML-DL-implementation/`
  to get a local copy of the source code in your environment.

- Install dependencies: You can use pip to install the dependendies on your computer.
  To install use
  `pip install -r requirements.txt`

- Installation:
  use `python setup.py develop` if you want to setup for development or `python setup.py install` if you only want to try and test out the repository.

- Make changes, work on a existing issue or create one. Once assigned you can start working on the issue.

- While you are working please make sure you follow standard programming guidelines. When you send us a PR, your code will be checked for PEP8 formatting and soon some tests will be added so that your code does not break already existing code. Use tools like flake8 to check your code for correct formatting.


# Algorithms Implemented

| Algorithm | Location |  Algorithm | Location | Algorithm | Location |
| :------------ | ------------: | :------------ | ------------: | :------------ | ------------: |
| **ACTIVATION FUNCTIONS**| |**OPTIMIZERS**|| **MODELS** | |
| Sigmoid |   [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L4) | Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L6) | Linear Regression | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L20) 
| Tanh | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L23) | StochasticGradientDescent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L58) | Logistic Regression| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L184) |
| Softmax | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L42) | Mini Batch Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L125) | Decision Tree Classifier| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L283)|
| Softsign | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L64) | Momentum Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L203) | KNN Classifier/Regessor| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L361) |
| Relu | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L102) | Nesterov Accelerated Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L296) | Naive Bayes | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L446)|
| Leaky Relu | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L102) | Adagrad | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L391) | Gaussian Naive Bayes| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L506) |
| Elu | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/activations.py#L121) | Adadelta | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L466) |  K Means Clustering| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/models.py#L560) |
| | | Adam | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/optimizers.py#L544) | 

| Algorithm | Location |
| :------------ | ------------: |
|**LOSS FUNCTIONS**| |
| Mean Squared Error | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/loss_func.py#L5)
|Log Error | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/loss_func.py#L57)
| Absolute Error | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/14d0afd4521e16b37c4011d02fd2aca8e6fdbd0e/MLlib/loss_func.py#L111)


