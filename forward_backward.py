import numpy as np
from scipy.special import logsumexp


# Alpha part
def next_log_alpha(prev_log_alpha, y, log_mu, log_trans):

    mu_term = log_mu[:, y]
    to_sumexp = prev_log_alpha.reshape(-1, 1) + log_trans
    sumexp_result = logsumexp(to_sumexp, axis=0)
    new_log_alpha = sumexp_result + mu_term

    return new_log_alpha


def compute_log_alphas(log_pi, log_mu, log_trans, ys):

    n_latent = log_pi.shape[0]

    log_alphas = np.zeros((ys.shape[0], n_latent))

    # Initial alpha
    log_alphas[0] = log_pi + log_mu[:, ys[0]]

    for i in range(1, ys.shape[0]):

        log_alphas[i] = next_log_alpha(log_alphas[i - 1], ys[i], log_mu,
                                       log_trans)

    return log_alphas


def previous_log_beta(log_beta, y, log_mu, log_trans):

    mu_term = log_mu[:, y]
    summed = log_beta.reshape(1, -1) + mu_term.reshape(1, -1) + log_trans
    prev_log_beta = logsumexp(summed, axis=1)

    return prev_log_beta


def compute_log_betas(log_mu, log_trans, ys):

    n_latent = log_trans.shape[0]

    log_betas = np.zeros((ys.shape[0], n_latent))
    log_betas[-1] = 0.

    for i in range(ys.shape[0] - 1, 0, -1):

        log_betas[i - 1] = previous_log_beta(log_betas[i], ys[i], log_mu,
                                             log_trans)

    return log_betas


def compute_log_xis(log_alphas, log_betas, log_mu, ys, log_trans, log_p):

    n_latent = log_trans.shape[0]

    log_xis = np.zeros((ys.shape[0] - 1, n_latent, n_latent))

    for i in range(ys.shape[0] - 1):

        log_xis[i] = (
            (log_alphas[i].reshape(-1, 1) + (
                log_mu[:, ys[i + 1]].reshape(1, -1) +
                log_betas[i + 1].reshape(1, -1)))
            + log_trans - log_p)

    return log_xis


def e_step(log_pi, log_mu, log_trans, ys):

    log_alphas = compute_log_alphas(log_pi, log_mu, log_trans, ys)

    log_betas = compute_log_betas(log_mu, log_trans, ys)

    log_p = logsumexp(log_alphas[-1])

    log_gammas = log_alphas + log_betas - log_p

    log_xis = compute_log_xis(log_alphas, log_betas, log_mu, ys, log_trans,
                              log_p)

    return log_gammas, log_xis, log_p
