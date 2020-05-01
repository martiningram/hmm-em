import numpy as np


def simulate_step(cur_state, B, trans_mat):

    n_obs = B.shape[1]
    n_latent = trans_mat.shape[0]

    cur_emission_probs = B[cur_state]
    cur_emission = np.random.choice(np.arange(n_obs), p=cur_emission_probs)

    next_state_probs = trans_mat[cur_state]
    next_state = np.random.choice(np.arange(n_latent), p=next_state_probs)

    return next_state, cur_emission


def simulate_hmm(init_probs, B, trans_mat, n_to_simulate):

    n_latent = trans_mat.shape[0]

    # Now we can simulate:
    init_z = np.random.choice(n_latent, p=init_probs)

    ys = np.zeros(n_to_simulate, dtype=int)
    zs = np.zeros(n_to_simulate, dtype=int)

    cur_z = init_z

    for i in range(n_to_simulate):

        zs[i] = cur_z

        cur_z, cur_y = simulate_step(cur_z, B, trans_mat)
        ys[i] = cur_y

    return ys, zs
