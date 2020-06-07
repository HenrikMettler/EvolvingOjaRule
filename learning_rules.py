import torch
import numpy as np
import matplotlib
import sklearn.datasets as datasets

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def initialize_weights(initial_weights, n):

    if initial_weights is None:
        initial_weights = np.random.uniform(-1, 1, size=n)

    # rescale the initial weights to squared sum 1
    return initial_weights / np.sqrt(np.sum(initial_weights ** 2))


def create_input_data(num_points, num_dimensions=2):
    """create an input data set for the learning rules

    Args:
        num_points: number of data points (m)
        num_dimensions (default: 2): data point dimension (n)

    Returns:
        input_data: m by n
         - m: Number of data points
         - n: Number of presynaptic inputs"""

    if num_points / num_dimensions <= 10:
        Warning(
            "The ration of num_points to num_dimensions is low, should be at least 10"
        )

    cov_mat = datasets.make_spd_matrix(num_dimensions)
    input_data = np.random.multivariate_normal(np.zeros(num_dimensions), cov_mat, num_points)

    return input_data


def learn_weights(input_data, learning_rule, initial_weights=None, learning_rate=0.005):
    """ learn weights
     Args:
        input_data (numpy array): An m by n array of datapoints.
            - m: Number of data points
            - n: Number of presynaptic inputs
        learning_rule (callable): determines the learning rule to be used. Current options:
            - 'oja_rule'
            - 'hebbian_rule'
            - 'normalized_hebbian_rule'
            - ('input_prompt_rule' not implemented yet)
        initial_weights (float, optional): initial weight vector
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        weights (numpy array): m by n, time course of the weight vector
        y (numpy array): time course of the output over learning"""

    m = np.size(input_data, 0)
    n = np.size(input_data, 1)

    w = initialize_weights(initial_weights, n)
    y = np.zeros([m, 1])  # initialize y
    weights = np.zeros([m, n])
    for i, x in enumerate(input_data):
        weights[i] = w
        y[i] = np.dot(w, x) # output: postsynaptic firing rate of a linear neuron
        w = learning_rule(w, x, y[i], learning_rate)

    return weights, y


def oja_rule(w, x, y, learning_rate=0.005):
    """Oja linearized learning rule (Oja, 1982: eq.3)

    Args:
        w (numpy_array): A 1 by n array of weights
        x (numpy array): A 1 by n array of presynaptic inputs .
            -n: Number of presynaptic inputs
        y (scalar): postsynaptic output
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        w_new (numpy array): 1 by n, updated weight vector
    """
    w_new = w + learning_rate * y * (x - y * w)

    return w_new


def normalized_hebbian_rule(w, x, y, learning_rate=0.005):
    """ normalized Hebbian rule (Oja, 1982: eq.2)

     Args:
        w (numpy_array): A 1 by n array of weights
        x (numpy array): A 1 by n array of presynaptic inputs .
            -n: Number of presynaptic inputs
        y (scalar): postsynaptic output
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        w_new (numpy array): 1 by n, updated weight vector"""


    scaling_factor = np.sqrt(
            np.sum(np.square(w + learning_rate * y * x))
        )
    w_new = (w + learning_rate * y * x) / scaling_factor

    return w_new


def hebbian_rule(w, x, y, learning_rate=0.005):
    """ Hebbian rule (eg Hebb 1961, Bishop 1995)

     Args:
        w (numpy_array): A 1 by n array of weights
        x (numpy array): A 1 by n array of presynaptic inputs .
            -n: Number of presynaptic inputs
        y (scalar): postsynaptic output
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        w_new (numpy array): 1 by n, updated weight vector"""

    w_new = (w + learning_rate * y * x)

    return w_new


def input_prompt_rule(w, x, y, learning_rate=0.005):
    """ Custom learning rule defined by the User (via imput prompt)

         Args:
            w (numpy_array): A 1 by n array of weights
            x (numpy array): A 1 by n array of presynaptic inputs .
                -n: Number of presynaptic inputs
            y (scalar): postsynaptic output
            learning_rate (float, optional): learning rate, default 0.005

        Returns:
            w_new (numpy array): 1 by n, updated weight vector"""

    temp_delta_w = input('Define your equation for delta_w')

    # Todo: assertions that temp_delta is valid (Criteria: dimensionality,...)
    is_valid = False
    if np.size(temp_delta_w) == np.size(w):
        is_valid = True
    assert is_valid
    delta_w = temp_delta_w
    w_new = w + delta_w

    return w_new


def plot_data(input_data, weights, y, do_data_plot=False):
    """
    Plots the amplitude of the angle between weights and PC1 of the input data,
    Plots a view of the data and the evolving axis, if do_dataPlot = true and m = 2

    Args:
        input_data (numpy.ndarray): m by n data
        weights (numpy.ndarray): m by n weights
        y(numpy.ndarray): m by 1 output values
        do_data_plot: boolean, whether to plot 2-dim data or not
    """

    # calculate the PC's of the input data (note: this could be placed out of the function also)
    pca = PCA(n_components=1)
    pca.fit(input_data)
    true_pc1 = pca.components_

    # plot the summed squared diff (weights - PC1)
    diff_w_pc1 = np.zeros([np.size(weights, 0)])
    for idx in range(np.size(weights, 0)):
        diff_w_pc1[idx] = np.sqrt(np.sum(np.square(weights[idx, :] - true_pc1)))

    y_min = 0.9 * np.min(diff_w_pc1)
    y_max = 1.1 * np.max(diff_w_pc1)
    plt.figure(1)
    plt.plot(diff_w_pc1)
    plt.ylim([y_min, y_max])
    plt.xlabel("sample")
    plt.ylabel("sq diff (w,PC1)")
    plt.title("Diff w, PC1")
    plt.show()

    # plot the angle between the weights vector and the PC1
    angles = np.zeros([np.size(weights, 0)])
    for idx in range(np.size(weights, 0)):
        current_weight = weights[idx] / np.linalg.norm(weights[idx])
        temp_dot_prod = np.dot(true_pc1, current_weight)
        angles[idx] = np.arccos(temp_dot_prod)
    piLine = 3.14 * np.ones([np.size(weights, 0)])
    plt.figure(2)
    plt.plot(angles)
    plt.plot(piLine)
    plt.ylim([0, 4])
    plt.xlabel("sample")
    plt.ylabel("angle in rad")
    plt.title("Angle between w, PC1 in rad")
    plt.legend(["angles", "pi"])
    plt.show()

    # plot the (accumulated) variance in y
    acc_var_y = np.zeros([np.size(y)])
    for idx in range(1, np.size(y, 0)):
        acc_var_y[idx] = np.var(y[0:idx])

    plt.figure(3)
    plt.plot(acc_var_y)
    plt.ylim([0, 1.2 * np.max(acc_var_y)])
    plt.xlabel("sample")
    plt.ylabel("variance")
    plt.title("Accumulated variance")
    plt.show()

    # plot the data and the final weight vector if desired and n_dim == 2
    if do_data_plot:
        if np.size(input_data, 1) != 2:
            print("Can not do data plot for n != 2")
            do_data_plot = False
        else:
            plt.scatter(
                input_data[:, 0],
                input_data[:, 1],
                marker=".",
                facecolor="none",
                edgecolor="#222222",
                alpha=0.2,
            )
            plt.xlabel("x1")
            plt.ylabel("x2")

            # color time and plot with colorbar
            time = np.arange(len(weights))
            colors = plt.cm.cool(time / float(len(time)))
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.cool, norm=plt.Normalize(vmin=0, vmax=len(input_data))
            )
            sm.set_array(time)
            cb = plt.colorbar(sm)
            cb.set_label("Iteration")
            plt.scatter(
                weights[:, 0], weights[:, 1], facecolor=colors, edgecolor="none", lw=2
            )

            # ensure rectangular plot
            x_min = input_data[:, 0].min()
            x_max = input_data[:, 0].max()
            y_min = input_data[:, 1].min()
            y_max = input_data[:, 1].max()
            lims = [min(x_min, y_min), max(x_max, y_max)]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.show()


def run(num_points, learning_rule, num_dimensions=2, initial_weights=None, learning_rate=0.005):

    input_data = create_input_data(num_points, num_dimensions)
    [weights, y] = learn_weights(input_data, learning_rule=learning_rule)
    plot_data(input_data, weights, y)


if __name__ == "__main__":
    np.random.seed(11)

    learning_rule = normalized_hebbian_rule
    run(num_points=10000, learning_rule=learning_rule, num_dimensions=10)
