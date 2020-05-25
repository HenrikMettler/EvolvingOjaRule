import torch
import numpy as np
import matplotlib.pyplot as plt

# Todo: implement something similar to angle, when dealing with n_dim > 2
def learn_oja(input_data, initial_weights=None, learning_rate = 0.005):
    """Implement Oja linearized learning rule

    Args:
        input_data (torch tensor): An M by N array of datapoints.
            - M: Number of data points
            - N: Number of presynaptic inputs
        initial_weights (float, optional): initial weight vector
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        weights (torch tensor): time course of the weight vector
        y (torch tensor): time course of the output over learning"""
    m = np.size(input_data,0)
    n = np.size(input_data,1)

    # set initial_weight if none
    if initial_weights is None:
        initial_weights = 2* np.random.rand(n,1) - 1 # random values between -1,1
    initial_weights = initial_weights / np.sum(np.square(initial_weights)) # rescale the initial weights to squared sum 1


    y = np.zeros([m,1])# initialize y
    weights = np.zeros(m,n)
    w = initial_weights
    for i in range(0, len(input_data)):
        weights[i] = w
        y[i] = np.dot(w, input_data[i])  # output: postsynaptic firing rate of a linear neuron.
        w += learning_rate * y[i] * (input_data[i] - y[i] * w)
    return weights, y


def learn_normalized_hebbian(input_data, initial_weights=None, learning_rate = 0.005):
    """ learn with the normalized hebbian rule (Oja-paper equation 2)

     Args:
        input_data (torch tensor): An M by N array of datapoints.
            - M: Number of data points
            - N: Number of presynaptic inputs
        initial_weights (float, optional): initial weight vector
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        weights (torch tensor): time course of the weight vector
        y (torch tensor): time course of the output over learning"""

def create_input_data(num_points, num_dimensions = 2, stds=None):
    """create an input data set for the learning rules

    Args:
        num_points: number of data points (m)
        stds (optional): std values along the different directions
        num_dimensions (default: 2): data point dimension (n)

    Returns:
        dataset of desired dimensionality """

    if num_points/num_dimensions <= 5:
        Warning('The ration of num_points to num_dimensions is low, should be at least 5')

    if stds is None:
        stds = np.zeros(num_dimensions,1)
        for idx_dim in range(num_dimensions):
            stds[idx_dim] = np.random.randn()
        # Todo: Rotate the axis

    input_data = np.random()
    return input_data

