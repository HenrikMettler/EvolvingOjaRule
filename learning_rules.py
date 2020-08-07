import numpy as np
import matplotlib

#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from functions import *


def oja_rule(w: np.ndarray, x: np.ndarray, y, learning_rate=0.005) -> np.ndarray:
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


def normalized_hebbian_rule(w: np.ndarray, x: np.ndarray, y, learning_rate=0.005) -> np.ndarray:
    """ normalized Hebbian rule (Oja, 1982: eq.2)

     Args:
        w (numpy_array): A 1 by n array of weights
        x (numpy array): A 1 by n array of presynaptic inputs .
            -n: Number of presynaptic inputs
        y (scalar): postsynaptic output
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        w_new (numpy array): 1 by n, updated weight vector"""

    scaling_factor = np.sqrt(np.sum(np.square(w + learning_rate * y * x)))
    w_new = (w + learning_rate * y * x) / scaling_factor

    return w_new


def hebbian_rule(w: np.ndarray, x: np.ndarray, y, learning_rate=0.005) -> np.ndarray:
    """ Hebbian rule (eg Hebb 1961, Bishop 1995)

     Args:
        w (numpy_array): A 1 by n array of weights
        x (numpy array): A 1 by n array of presynaptic inputs .
            -n: Number of presynaptic inputs
        y (scalar): postsynaptic output
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        w_new (numpy array): 1 by n, updated weight vector"""

    w_new = w + learning_rate * y * x

    return w_new


def plot_data(input_data: np.ndarray, weights: np.ndarray, y: np.ndarray, do_data_plot=False) -> None:
    """
    Plots the amplitude of the angle between weights and PC1 of the input data,
    Plots a view of the data and the evolving axis, if do_dataPlot = true and m = 2

    Args:
        input_data (numpy.ndarray): m by n data
        weights (numpy.ndarray): m by n weights
        y(numpy.ndarray): m by 1 output values
        do_data_plot: boolean, whether to plot 2-dim data or not
    """

    true_pc1 = compute_first_pc(input_data)
    diff_w_pc1 = compute_difference_weights_first_pc(weights, true_pc1)
    angles_w_pc1 = compute_angles_weights_first_pc(weights, true_pc1)
    acc_var_y = compute_accumulated_variance(y)

    # plot the mean squared difference
    y_min = 0.9 * np.min(diff_w_pc1)
    y_max = 1.1 * np.max(diff_w_pc1)
    plt.figure(1)
    plt.plot(diff_w_pc1)
    plt.ylim([y_min, y_max])
    plt.xlabel("sample")
    plt.ylabel("sq diff (w,PC1)")
    plt.title("Diff w, PC1")
    plt.show()

    # plot the angle between weight and pc1
    pi_line = 3.14 * np.ones([np.size(weights, 0)])
    plt.figure(2)
    plt.plot(angles_w_pc1)
    plt.plot(pi_line)
    plt.ylim([0, 4])
    plt.xlabel("sample")
    plt.ylabel("angle in rad")
    plt.title("Angle between w, PC1 in rad")
    plt.legend(["angles", "pi"])
    plt.show()

    # plot the (accumulated) variance in y
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
            raise NotImplementedError("Can not do data plot for n != 2")
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


def run(
    num_points,
    learning_rule,
    num_dimensions=2,
    initial_weights=None,
    learning_rate=0.005,
) -> None:

    input_data, _ = create_input_data(num_points, num_dimensions)
    [weights, y] = learn_weights(input_data, learning_rule=learning_rule)
    plot_data(input_data, weights, y)


if __name__ == "__main__":
    np.random.seed(11)

    learning_rule = oja_rule
    run(num_points=10000, learning_rule=learning_rule, num_dimensions=10)
