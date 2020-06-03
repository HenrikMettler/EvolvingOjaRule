import torch
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def initialize_weights(initial_weights, n):

    if initial_weights is None:
        initial_weights = np.random.uniform(-1, 1, size=n)

    # rescale the initial weights to squared sum 1
    return initial_weights / np.sqrt(np.sum(initial_weights ** 2))


def sim(input_data, initial_weights, learning_rate, plasticity_rule):
    """
    Args:
        input_data (numpy array): An m by n array of datapoints.
            -m: Number of data points
            -n: Number of presynaptic inputs
        initial_weights (float): initial weight vector
        learning_rate (float): learning rate
        plasticity_rule (callable): plasticity rule

    Returns:
        weights (numpy array): m by n, time course of the weight vector
        y (numpy array): time course of the output over learning
    """

    m = np.size(input_data, 0)
    n = np.size(input_data, 1)

    w = initialize_weights(initial_weights, n)
    y = np.zeros([m, 1])
    weights = np.zeros([m, n])
    for i, x in enumerate(input_data):
        weights[i] = w
        y[i] = np.dot(w, x)
        w = plasticity_rule(w, y[i], x)

    return weights, y


def learn_oja(input_data, initial_weights=None, learning_rate=0.005):
    """Oja's linearized learning rule (Oja, 1982; eq 3)"""

    def plasticity_rule(w, y, x):
        return w + learning_rate * y * (x - y * w)

    return sim(input_data, initial_weights, learning_rate, plasticity_rule)


def learn_normalized_hebbian(input_data, initial_weights=None, learning_rate=0.005):
    """Normalized Hebbian rule (Oja, 1982; eq 2)"""

    def plasticity_rule(w, y, x):
        pass  # TODO implement

    return sim(input_data, initial_weights, learning_rate, plasticity_rule)


def create_input_data(num_points, num_dimensions=2, stds=None):
    """create an input data set for the learning rules

    Args:
        num_points: number of data points (m)
        stds (optional): std values along the different directions
        num_dimensions (default: 2): data point dimension (n)

    Returns:
        input_data: m by n
         - m: Number of data points
         - n: Number of presynaptic inputs"""

    if num_points / num_dimensions <= 10:
        Warning(
            "The ration of num_points to num_dimensions is low, should be at least 10"
        )

    if stds is None:
        stds = np.random.randn(num_dimensions)
        stds = np.abs(stds)

    return np.random.normal(0, stds, size=[num_points, num_dimensions])


def compute_first_principle_component(input_data):
    pca = PCA(n_components=1)
    pca.fit(input_data)
    return pca.components_[0]


def compute_difference_between_weights_and_first_principle_component(weights, pc1):
    diff_w_pc1 = np.zeros(len(weights))
    for idx in range(len(weights)):
        diff_w_pc1[idx] = np.linalg.norm(weights[idx] - pc1)
    return diff_w_pc1


def compute_angle_between_weights_and_first_principle_component(weights, pc1):
    assert abs(np.linalg.norm(pc1) - 1.0) < 1e-10

    angles = np.zeros(len(weights))
    for idx in range(len(weights)):
        current_normalized_weight = weights[idx] / np.linalg.norm(weights[idx])
        dot_prod = np.dot(pc1, current_normalized_weight)
        angles[idx] = np.arccos(dot_prod)
    return angles


def compute_accumulated_variance(y):
    acc_var_y = np.zeros(len(y))
    for idx in range(1, len(y)):
        acc_var_y[idx] = np.var(y[:idx])
    return acc_var_y


def plot_oja(input_data, weights, y, plot_input_data=False):
    """
    Plots the amplitude of the angle between weights and PC1 of the input data,
    Plots a view of the data and the evolving axis, if plot_input_data = true and m = 2

    Args:
        input_data (numpy.ndarray): m by n data
        weights_course (numpy.ndarray): m by n weights
        y(numpy.ndarray): m by 1 output values

    """

    pc1 = compute_first_principle_component(input_data)
    diff_w_pc1 = compute_difference_between_weights_and_first_principle_component(
        weights, pc1
    )
    angles_w_pc1 = compute_angle_between_weights_and_first_principle_component(
        weights, pc1
    )
    acc_var_y = compute_accumulated_variance(y)

    plt.figure(1)
    plt.plot(diff_w_pc1)
    plt.ylim([0.0, 5.0])
    plt.xlabel("sample")
    plt.ylabel("sq diff (w,PC1)")
    plt.title("Diff w, PC1")
    plt.show()

    plt.figure(2)
    plt.plot(angles_w_pc1, color="C0")
    plt.axhline(np.pi, color="C1")
    plt.ylim([0, 3.5])
    plt.xlabel("sample")
    plt.ylabel("angle in rad")
    plt.title("Angle between w, PC1 in rad")
    plt.legend(["angles", "pi"])
    plt.show()

    plt.figure(3)
    plt.plot(acc_var_y)
    plt.ylim([0, 1.2 * np.max(acc_var_y)])
    plt.xlabel("sample")
    plt.ylabel("variance")
    plt.title("Accumulated variance")
    plt.show()

    if plot_input_data:
        if np.size(input_data, 1) != 2:
            raise NotImplementedError(
                "Can not plot input data for input dimension != 2"
            )

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


def run_oja(
    num_points,
    num_dimensions=2,
    stds=None,
    initial_weights=None,
    learning_rate=0.005,
    plot_input_data=False,
):

    input_data = create_input_data(num_points, num_dimensions, stds)
    weights, y = learn_oja(input_data, initial_weights, learning_rate)

    plot_oja(input_data, weights, y, plot_input_data)


if __name__ == "__main__":
    np.random.seed(12)  # always make results reproducible

    run_oja(num_points=5000, num_dimensions=2, plot_input_data=True)
