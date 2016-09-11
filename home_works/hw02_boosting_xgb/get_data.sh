#!/bin/bash

# ==== Download adult ====
curl -o ./data/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# ==== Download cifar10 ====
mkdir ./data/cifar10
curl -o ./data/cifar10/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf ./data/cifar10/cifar-10-python.tar.gz -C ./data/cifar10/
