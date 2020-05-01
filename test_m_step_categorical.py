import numpy as np
from forward_backward import e_step
from test_forward_backward import generate_random_params_and_data
from m_step_categorical import log_a_update, log_mu_update


def test_log_a_update():

    obs, mu, trans_mat, pi = generate_random_params_and_data()

    log_gammas, log_xis, log_p = e_step(
        np.log(pi), np.log(mu), np.log(trans_mat), obs)

    xis = np.exp(log_xis)

    # Xis are of shape [n_obs, n_latent, n_latent].
    numerator = np.sum(xis, axis=0)
    denominator = np.sum(np.sum(xis, axis=0), axis=1, keepdims=True)

    new_a = numerator / denominator

    alt_a = log_a_update(log_xis)

    assert np.allclose(np.exp(alt_a), new_a)


def test_log_mu_update():

    obs, mu, trans_mat, pi = generate_random_params_and_data()

    log_gammas, log_xis, log_p = e_step(
        np.log(pi), np.log(mu), np.log(trans_mat), obs)

    n_latent = trans_mat.shape[0]
    n_outputs = mu.shape[1]

    new_log_mus = log_mu_update(obs, log_gammas, mu.shape[1])

    numerators = np.zeros((n_outputs, n_latent))

    # Compute numerators
    for n in range(obs.shape[0]):
        for cur_output in range(mu.shape[1]):
            for k in range(n_latent):

                # Numerator
                cur_numerator = np.exp(
                    log_gammas[n, k]) * (obs[n] == cur_output)

                numerators[cur_output, k] += cur_numerator

    # Compute denominators
    denominators = np.sum(np.exp(log_gammas), axis=0)

    alt_result = numerators / denominators.reshape(1, -1)

    alt_result = alt_result.T

    assert np.allclose(new_log_mus, np.log(alt_result))
