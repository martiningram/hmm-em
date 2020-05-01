import numpy as np
from scipy.special import logsumexp


def log_pi_update(log_gamma_1):

    return log_gamma_1 - logsumexp(log_gamma_1)


def log_a_update(log_xis):

    A_top_log = logsumexp(log_xis, axis=0)
    A_bottom_log = logsumexp(log_xis, axis=(0, 2))
    new_log_trans = A_top_log - A_bottom_log.reshape(-1, 1)

    return new_log_trans


def log_mu_update(ys, log_gammas, n_outputs):

    n_latent = log_gammas.shape[1]

    x_ni = np.zeros((ys.shape[0], n_outputs))
    x_ni[np.arange(ys.shape[0]), ys] = 1

    new_log_mu = np.zeros((n_latent, n_outputs))

    for cur_out in range(n_outputs):

        cur_filter = x_ni[:, cur_out] == 1
        cur_log_gammas = log_gammas[cur_filter]
        cur_summed = logsumexp(cur_log_gammas, axis=0)
        cur_denom = logsumexp(log_gammas, axis=0)

        new_log_mu[:, cur_out] = cur_summed - cur_denom

    return new_log_mu


def m_step(log_gammas, log_xis, ys, n_outputs):

    new_log_pi = log_pi_update(log_gammas[0])
    new_log_trans = log_a_update(log_xis)
    new_log_mu = log_mu_update(ys, log_gammas, n_outputs)

    return new_log_pi, new_log_trans, new_log_mu
