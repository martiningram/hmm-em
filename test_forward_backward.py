import numpy as np
from scipy.special import softmax, logsumexp
from forward_backward import (
    next_log_alpha, previous_log_beta, compute_log_alphas, compute_log_betas,
    compute_log_xis)


def generate_random_params_and_data():

    np.random.seed(2)

    n_latent = 3

    obs = np.array([0, 1])

    mu = softmax(np.random.randn(3, 2), axis=1)
    trans_mat = softmax(np.random.randn(3, 3), axis=1)

    # Start with a distribution pi
    pi = np.repeat(1 / n_latent, n_latent)

    return obs, mu, trans_mat, pi


def test_next_log_alpha():

    obs, mu, trans_mat, pi = generate_random_params_and_data()

    # Initial distribution:
    alpha_0 = pi * mu[:, obs[0]]

    sum_part = alpha_0.reshape(1, -1) @ trans_mat
    alpha_1 = mu[:, obs[1]] * sum_part

    log_alpha_1 = next_log_alpha(
        np.log(alpha_0), obs[1], np.log(mu), np.log(trans_mat))

    assert np.allclose(log_alpha_1, np.log(alpha_1))

    # Also check log betas:
    log_alphas = compute_log_alphas(np.log(pi), np.log(mu), np.log(trans_mat),
                                    obs)

    assert np.allclose(np.exp(log_alphas[0]), alpha_0.reshape(-1))
    assert np.allclose(np.exp(log_alphas[1]), alpha_1.reshape(-1))


def test_previous_log_beta():

    obs, mu, trans_mat, pi = generate_random_params_and_data()

    n_latent = trans_mat.shape[0]

    # Start at the end with beta all ones
    beta_z2 = np.ones(n_latent)

    # Compute beta_z1 from this
    multiplied = beta_z2 * mu[:, obs[1]]
    beta_z1 = trans_mat @ multiplied.reshape(-1, 1)

    log_beta_z1 = previous_log_beta(
        np.log(beta_z2), obs[1], np.log(mu), np.log(trans_mat))

    assert np.allclose(log_beta_z1, np.log(beta_z1).reshape(-1))

    # Also check log betas:
    log_betas = compute_log_betas(np.log(mu), np.log(trans_mat), obs)

    assert np.allclose(np.exp(log_betas[0]), beta_z1.reshape(-1))
    assert np.allclose(np.exp(log_betas[1]), beta_z2.reshape(-1))


def test_calculate_log_xis():

    obs, mu, trans_mat, pi = generate_random_params_and_data()

    n_obs = obs.shape[0]
    n_latent = trans_mat.shape[0]

    log_alphas = compute_log_alphas(
        np.log(pi), np.log(mu), np.log(trans_mat), obs)

    log_betas = compute_log_betas(np.log(mu), np.log(trans_mat), obs)

    alphas, betas = map(np.exp, [log_alphas, log_betas])

    p_x = np.sum(alphas[-1])

    xis = np.zeros((n_obs - 1, n_latent, n_latent))

    # Compute the first xi
    xis[0] = np.outer(alphas[0], betas[1] * mu[:, obs[1]]) * trans_mat / p_x

    log_xis = compute_log_xis(log_alphas, log_betas, np.log(mu), obs,
                              np.log(trans_mat), np.log(p_x))

    assert np.allclose(np.exp(log_xis), xis)


def test_calculate_log_alpha():

    # Just a sanity check
    alpha = np.random.uniform(0.0001, 1., size=3)

    assert np.log(np.sum(alpha)) == logsumexp(np.log(alpha))
