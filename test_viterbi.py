import numpy as np
from viterbi import viterbi, viterbi_log


def test_viterbi_against_log_viterbi():

    A = np.array([[0.6, 0.4],
                  [0.5, 0.5]])

    B = np.array([[0.2, 0.4, 0.4],
                  [0.5, 0.4, 0.1]])

    obs_seq = np.array([0, 2, 0, 1, 0, 1, 0, 0, 0])

    pi = np.array([0.8, 0.2])

    deltas, a_s, path = viterbi(pi, A, B, obs_seq)

    log_deltas, log_a_s, log_path = viterbi_log(
        np.log(pi), np.log(A), np.log(B), obs_seq)

    assert np.allclose(path, log_path)
