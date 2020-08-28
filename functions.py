import numpy as np
import sklearn.datasets as sklearn_datasets
import warnings
import scipy

from sklearn.decomposition import PCA
from sympy import symbols
from sympy.utilities.lambdify import lambdify


def initialize_weights(n, rng):

    initial_weights = rng.uniform(-1, 1, size=n)
    return initial_weights / np.sqrt(np.sum(initial_weights ** 2))  # rescale to squared sum 1


def create_input_data(num_points, num_dimensions, max_var_input, seed, data_mean = 0):
    """create an input data set for the learning rules

    Args:
        num_points: number of data points (m)
        num_dimensions (default: 2): data point dimension (n)
        max_var_input: gives the maximum input variance in input
        seed:
    Returns:
        input_data: m by n
         - m: Number of data points
         - n: Number of presynaptic inputs"""

    if num_points / num_dimensions <= 10:
        Warning(
            "The ration of num_points to num_dimensions is low, should be at least 10"
        )

    cov_mat = sklearn_datasets.make_spd_matrix(num_dimensions, seed)
    # rescale the cov_mat to have max_var_input as maximal element
    cov_mat = max_var_input * cov_mat / np.max(cov_mat)
    input_data = np.random.multivariate_normal(
        data_mean*np.ones(num_dimensions), cov_mat, num_points)
    return input_data, cov_mat


def calculate_eigenvector_for_largest_eigenvalue(cov_mat: np.ndarray) -> np.ndarray:
    n_dim = np.size(cov_mat, 0)
    eigenvalue, eigenvector = scipy.linalg.eigh(cov_mat, subset_by_index=[n_dim-1, n_dim-1])
    eigenvector = np.reshape(eigenvector, (n_dim,))  # needed so that eigenvectors are of same shape as weight vectors
    return eigenvector


def learn_weights(input_data, learning_rule, initial_weights, learning_rate=0.005):
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
        initial_weights (float): initial weight vector
        learning_rate (float, optional): learning rate, default 0.005

    Returns:
        weights (numpy array): m by n, time course of the weight vector
        y (numpy array): time course of the output over learning"""

    m = np.size(input_data, 0)
    n = np.size(input_data, 1)

    w = np.ndarray.copy(initial_weights)
    y = np.zeros([m, 1])  # initialize y
    weights = np.zeros([m, n])
    for i, x in enumerate(input_data):
        weights[i] = w
        y[i] = np.dot(w, x)  # output: postsynaptic firing rate of a linear neuron
        try:  # this works for the standard learning rules (Oja, (norm) Hebb)
            w = learning_rule(w, x, y[i], learning_rate)
        except TypeError:
            # calculate the weight update for every weight separately from the cartesian graph
            # todo: can be done in vectors with .to_numpy function!
            with warnings.catch_warnings():  # Todo: why they arise & if all caught
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in reduce "
                    "arrmean = umr_sum(arr, axis, dtype, keepdims=True)",
                )
                warnings.filterwarnings(
                    "ignore", message="overflow encountered in double_scalars"
                )
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in double_scalars"
                )
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in subtract"
                )
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in multiply"
                )
                warnings.filterwarnings(
                    "ignore", message="overflow encountered in multiply"
                )
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in add"
                )
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in reduce"
                )

                for idx in range(n):
                    graph_in = np.array([[x[idx], w[idx], y[i]]])  #
                    graph_out = learning_rule(graph_in)
                    # Catch if the graph returns nan, set final weights all to nan and return
                    if np.isnan(graph_out[0][0]):
                        weights[-1] = graph_out[0][0]
                        return weights, y
                    w[idx] += (
                        learning_rate * graph_out
                    )

    return weights, y


def learn_weights_with_lambdify(dataset, learning_rule_as_lambdify, initial_weights, learning_rate):
    m = np.size(dataset, 0)
    n = np.size(dataset, 1)

    w = np.ndarray.copy(initial_weights)
    y = np.zeros([m, 1])
    weights = np.zeros([m, n])
    for i, x in enumerate(dataset):
        weights[i] = w
        y[i] = np.dot(w, x)
        for idx in range(n):
            lambdify_out = learning_rule_as_lambdify(x[idx], w[idx], y[i])
            # Catch if the graph returns nan, set final weights all to nan and return
            w[idx] += learning_rate * lambdify_out

    return weights, y


def evaluate_output(evaluation_data, weights):
    m = np.size(evaluation_data, 0)
    y = np.zeros([m, 1])  # initialize y
    for i in range(m):
        y[i] = np.dot(
            weights, evaluation_data[i]
        )  # output: postsynaptic firing rate of a linear neuron
    return y


def calc_weight_penalty(weights, mode):
    if mode == 1:
        out = np.square(np.sqrt(np.sum(np.square(weights))) - 1)
    elif mode == 0:
        out = np.sqrt(np.sum(np.square(weights)))
    else:
        raise NotImplementedError(
            "Mode for calculating the weight penalty not available, choose 0 or 1"
        )
    return out


def compute_first_pc(input_data):
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        print("oops")
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(input_data)
    return pca.components_[0]


def compute_difference_weights_first_pc(weights, pc1):
    diff_w_pc1 = np.zeros(len(weights))
    for idx in range(len(weights)):
        diff_w_pc1[idx] = np.sqrt(np.sum(np.square(weights[idx, :] - pc1)))
    return diff_w_pc1


def compute_angle_weight_first_pc(weight, pc0, mode="rad"):
    weight_rescale = weight / np.linalg.norm(weight)
    dot_prod = np.dot(pc0, weight_rescale)
    angle = np.arccos(dot_prod)
    if mode == "rad":
        return angle
    elif mode == "degree":
        return angle * 360 / (np.pi *2)


def compute_angles_weights_first_pc(weights, pc0):
    angles = np.zeros(len(weights))
    for idx in range(np.size(weights, 0)):
        angles[idx] = compute_angle_weight_first_pc(weights[idx], pc0)
    return angles


def calculate_smallest_angle(angle: float) -> float:
    min_360 = np.min([angle, 360 - angle])
    smallest_angle = np.min([min_360, abs(angle - 180)])
    return smallest_angle


def calc_difference_to_first_pc(input_data, weight):
    # calc PC1 of dataset
    data_first_pc = compute_first_pc(input_data)
    # calc angle PC1 to w
    angle = compute_angle_weight_first_pc(weight, data_first_pc, mode="rad")
    # calc diff to Pi
    first_term = np.minimum(angle, np.pi - angle)
    return first_term


def compute_accumulated_variance(y):
    acc_var_y = np.zeros([np.size(y)])
    for idx in range(1, np.size(y, 0)):
        acc_var_y[idx] = np.var(y[0:idx])
    return acc_var_y


def replace_expression(champion_expression):
    rep_x0_x = champion_expression.subs('x_0', 'x')
    rep_x1_w = rep_x0_x.subs('x_1', 'w')
    learning_rule_expression = rep_x1_w.subs('x_2', 'y')
    return learning_rule_expression


def create_function_from_expression(learning_rule_expression):
    x = symbols('x')
    w = symbols('w')
    y = symbols('y')
    learning_rule_as_function = lambdify([x, w, y], learning_rule_expression)
    return learning_rule_as_function