import autograd.numpy as np
import celerite
from celerite import terms

class MixtureOfSHOsTerm(terms.SHOTerm):
    parameter_names = ("log_a", "log_Q1", "mix_par", "log_Q2", "log_P")

    def __repr__(self):
        return "MixtureOfSHOserm({0.log_a}, {0.log_Q1}, {0.mix_par}, {0.log_Q2}, {0.log_P})".format(self)

    def get_real_coefficients(self, params):
        return np.empty(0), np.empty(0)

    def get_complex_coefficients(self, params):
        log_a, log_Q1, mix_par, log_Q2, log_period = params

        Q = np.exp(log_Q2) + np.exp(log_Q1)
        log_Q1 = np.log(Q)
        P = np.exp(log_period)
        log_omega1 = np.log(4*np.pi*Q) - np.log(P) - 0.5*np.log(4.0*Q*Q-1.0)
        log_S1 = log_a - log_omega1 - log_Q1

        mix = -np.log(1.0 + np.exp(-mix_par))
        Q = np.exp(log_Q2)
        P = 0.5*np.exp(log_period)
        log_omega2 = np.log(4*np.pi*Q) - np.log(P) - 0.5*np.log(4.0*Q*Q-1.0)
        log_S2 = mix + log_a - log_omega2 - log_Q2

        c1 = super(MixtureOfSHOsTerm, self).get_complex_coefficients([
            log_S1, log_Q1, log_omega1,
        ])

        c2 = super(MixtureOfSHOsTerm, self).get_complex_coefficients([
            log_S2, log_Q2, log_omega2,
        ])

        return [np.array([a, b]) for a, b in zip(c1, c2)]

    def log_prior(self):
        lp = super(MixtureOfSHOsTerm, self).log_prior()
        if not np.isfinite(lp):
            return -np.inf
        mix = 1.0 / (1.0 + np.exp(-self.mix_par))
        return lp + np.log(mix) + np.log(1.0 - mix)