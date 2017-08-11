# active_testing
This is a python package for testing controllers for black box system in simulators

# Installation
To install, 
```
sudo python setup.py install
```
This should allow you to use the package anywhere in your current environment

# Tests
The tests folder has 3 files:
- test_sincos.py : This file shows the difference between modeling smooth and non-smooth functions using GPy
- test_car.py: This file implements a simple linear controller on a car for obstacle avoidance. This file shows how we use KernelPCA for reducing the input space.
- test_cartpole.py: This file tests a nearest neighbor controllder code submitted by user for the Open AI Gym environment Cartpole-v0
