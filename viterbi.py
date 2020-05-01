import numpy as np


def viterbi(pi, A, B, obs_sequence):
    # TODO: This should be in log space!
    # Takes:
    # pi: Initial distribution over latent states
    # A: Transition matrix.
    # B: Hidden-to-observed matrix
    # obs_sequence: Observed states (categorial here, so a sequence of
    # integers)

    n_obs = len(obs_sequence)
    n_hidden = A.shape[0]

    # Phi as defined in Murphy
    phi = B[:, obs_sequence].T

    # These will store the delta_t(j) and a_t(j)
    deltas = np.zeros((n_obs, n_hidden))
    a_s = np.zeros((n_obs, n_hidden))

    # Start off with delta at time 1 (not we're using python which starts
    # indices at zero)
    deltas[0] = pi * phi[0]

    # Do the recursion
    for i in range(1, n_obs):

        # Pick out the current phi -- this is a vector giving
        # the likelihood of each hidden state given the observation
        cur_phi = phi[i]

        # Compute delta
        cur_full_delta = np.outer(deltas[i - 1], cur_phi) * A
        cur_delta = np.max(cur_full_delta, axis=0)
        deltas[i] = cur_delta

        # Compute a
        a_s[i] = np.argmax(cur_full_delta, axis=0)

    # Back-tracking:
    z_stars = np.zeros(n_obs, dtype=np.int)

    # Most probable final state:
    z_stars[-1] = deltas[-1].argmax()

    # Note tricky indexing: start at n_obs - 2 because n_obs - 1
    # is the last element and we want one before that.
    for i in range(n_obs - 2, 0, -1):

        z_stars[i] = a_s[i + 1, z_stars[i + 1]]

    # Return the results -- don't really need the a_s but hey
    return deltas, a_s, z_stars


def viterbi_log(log_pi, log_A, log_B, obs_sequence):
    # Takes:
    # pi: Initial distribution over latent states
    # A: Transition matrix.
    # B: Hidden-to-observed matrix
    # obs_sequence: Observed states (categorial here, so a sequence of
    # integers)

    n_obs = len(obs_sequence)
    n_hidden = log_A.shape[0]

    # Phi as defined in Murphy
    log_phi = log_B[:, obs_sequence].T

    # These will store the delta_t(j) and a_t(j)
    log_deltas = np.zeros((n_obs, n_hidden))
    a_s = np.zeros((n_obs, n_hidden))

    # Start off with delta at time 1 (not we're using python which starts
    # indices at zero)
    log_deltas[0] = log_pi + log_phi[0]

    # Do the recursion
    for i in range(1, n_obs):

        # Pick out the current phi -- this is a vector giving
        # the likelihood of each hidden state given the observation
        cur_log_phi = log_phi[i]

        # Compute delta
        cur_full_log_delta = (log_deltas[i - 1].reshape(-1, 1) +
                              cur_log_phi.reshape(1, -1) + log_A)

        cur_log_delta = np.max(cur_full_log_delta, axis=0)

        log_deltas[i] = cur_log_delta

        # Compute a
        a_s[i] = np.argmax(cur_full_log_delta, axis=0)

    # Back-tracking:
    z_stars = np.zeros(n_obs, dtype=np.int)

    # Most probable final state:
    z_stars[-1] = log_deltas[-1].argmax()

    # Note tricky indexing: start at n_obs - 2 because n_obs - 1
    # is the last element and we want one before that.
    for i in range(n_obs - 2, 0, -1):

        z_stars[i] = a_s[i + 1, z_stars[i + 1]]

    # Return the results -- don't really need the a_s but hey
    return log_deltas, a_s, z_stars
