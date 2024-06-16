def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):

    if label * np.dot(theta, feature_vector) + theta_0 <= 1:
        theta = (1-eta*L) * theta + (eta*label*feature_vector)
        theta_0 = theta_0 + label * eta
    else:
        theta = (1-eta*L) * theta
        theta_0 = theta_0

    return theta, theta_0

def pegasos(feature_matrix, labels, T, L):

    m = feature_matrix.shape[0]
    n = feature_matrix.shape[1]
    theta = np.zeros((m,))
    theta_0 = 0.0
    all = [i for i in range(1, n * T + 1)]
    index = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):

            eta = 1 / np.sqrt(all[index])
            (theta, theta_0) = pegasos_single_step_update(
                feature_matrix[i, :],
                labels[i],
                L,
                eta,
                theta,
                theta_0)
            index += 1

    return theta, theta_0
