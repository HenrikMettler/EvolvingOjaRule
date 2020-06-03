"""From: https://neuronaldynamics-exercises.readthedocs.io/en/latest/_modules/neurodynex/ojas_rule/oja.html#learn
(NeuroDynEx source code) - OA: Gerstner, Kistler, Paninski, Naud"""

import matplotlib.pyplot as plt
import numpy as np


def make_cloud(n=2000, ratio=1, angle=0):
    """Returns an oriented elliptic
    gaussian cloud of 2D points

    Args:
        n (int, optional): number of points in the cloud
        ratio (int, optional): (std along the short axis) /
            (std along the long axis)
        angle (int, optional): rotation angle [deg]

    Returns:
        numpy.ndarray: array of datapoints
    """

    if ratio > 1.:
        ratio = 1. / ratio

    x = np.random.randn(n, 1)
    y = ratio * np.random.randn(n, 1)
    z = np.concatenate((x, y), 1)
    radangle = (180. - angle) * np.pi / 180.
    transfo = [
        [np.cos(radangle), np.sin(radangle)],
        [-np.sin(radangle), np.cos(radangle)]
    ]
    return np.dot(transfo, z.T).T



def learn(cloud, initial_angle=None, eta=0.005):
    """Run one batch of Oja's learning over
    a cloud of datapoints.

    Args:
        cloud (numpy.ndarray): An N by 2 array of datapoints. You can
            think of each of the two columns as the time series of firing rates of one presynaptic neuron.
        initial_angle (float, optional): angle of initial
            set of weights [deg]. If None, this is random.
        eta (float, optional): learning rate

    Returns:
        numpy.ndarray: time course of the weight vector
    """

    # get angle if not set
    if initial_angle is None:
        initial_angle = np.random.rand() * 360.
    radangle = initial_angle * np.pi / 180.

    w = np.array([np.cos(radangle), np.sin(radangle)])
    wcourse = np.zeros((len(cloud), 2), float)
    for i in range(0, len(cloud)):
        wcourse[i] = w
        y = np.dot(w, cloud[i])  # output: postsynaptic firing rate of a linear neuron.
        # ojas rule (cloud[i] are the two presynaptic firing rates at time point i
        w = w + eta * y * (cloud[i] - y * w)
    return wcourse


def plot_oja_trace(data_cloud, weights_course):
    """
    Plots the datapoints and the time series of the weights
    Args:
        data_cloud (numpy.ndarray): n by 2 data
        weights_course (numpy.ndarray): n by 2 weights

    Returns:

    """
    plt.scatter(
        data_cloud[:, 0],
        data_cloud[:, 1],
        marker=".",
        facecolor="none",
        edgecolor="#222222",
        alpha=.2
    )
    plt.xlabel("x1")
    plt.ylabel("x2")

    # color time and plot with colorbar
    time = np.arange(len(weights_course))
    colors = plt.cm.cool(time / float(len(time)))
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.cool,
        norm=plt.Normalize(vmin=0, vmax=len(data_cloud))
    )
    sm.set_array(time)
    cb = plt.colorbar(sm)
    cb.set_label("Iteration")
    plt.scatter(
        weights_course[:, 0],
        weights_course[:, 1],
        facecolor=colors,
        edgecolor="none",
        lw=2
    )

    # ensure rectangular plot
    x_min = data_cloud[:, 0].min()
    x_max = data_cloud[:, 0].max()
    y_min = data_cloud[:, 1].min()
    y_max = data_cloud[:, 1].max()
    lims = [min(x_min, y_min), max(x_max, y_max)]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.show()




def run_oja(n=2000, ratio=1., angle=0., learning_rate=0.01, do_plot=True):
    """Generates a point cloud and runs Oja's learning
    rule once. Optionally plots the result.

    Args:
        n (int, optional): number of points in the cloud
        ratio (float, optional): (std along the short axis) /
            (std along the long axis)
        angle (float, optional): rotation angle [deg]
        do_plot (bool, optional): plot the result
    """

    cloud = make_cloud(n=n, ratio=ratio, angle=angle)
    wcourse = learn(cloud, eta=learning_rate)

    if do_plot:
        plot_oja_trace(cloud, wcourse)
    return wcourse, cloud


if __name__ == "__main__":
    run_oja(n=2000, ratio=1.1, angle=30, learning_rate=0.2)