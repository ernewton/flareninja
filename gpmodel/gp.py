import autograd.numpy as np
import celerite
from celerite import terms

# -*- coding: utf-8 -*-



__all__ = ["get_simple_gp", "get_rotation_gp"]

import numpy as np

import celerite
from celerite import terms

from mixterm import MixtureOfSHOsTerm
# from .rotation_term import MixtureTerm


def get_basic_kernel(t, y, yerr):
    kernel = terms.SHOTerm(
        log_S0=np.log(np.var(y)),
        log_Q=-np.log(4.0),
        log_omega0=np.log(2*np.pi/10.),
        bounds=dict(
            log_S0=(-20.0, 10.0),
            log_omega0=(np.log(2*np.pi/80.0), np.log(2*np.pi/2.0)),
        ),
    )
    kernel.freeze_parameter('log_Q')

    # Finally some jitter
    kernel += terms.JitterTerm(log_sigma=np.log(np.median(yerr)),
                               bounds=[(-20.0, 5.0)])

    return kernel


def get_simple_gp(t, y, yerr):
    gp = celerite.GP(kernel=get_basic_kernel(t, y, yerr), mean=0.)
    gp.compute(t)
    return gp


def get_rotation_gp(t, y, yerr, period, min_period, max_period):
    kernel = get_basic_kernel(t, y, yerr)
    kernel += MixtureOfSHOsTerm(
        log_a=np.log(np.var(y)),
        log_Q1=np.log(15),
        mix_par=-1.0,
        log_Q2=np.log(15),
        log_P=np.log(period),
        bounds=dict(
            log_a=(-20.0, 10.0),
            log_Q1=(-0.5*np.log(2.0), 8.0),
            mix_par=(-5.0, 5.0),
            log_Q2=(-0.5*np.log(2.0), 8.0),
            log_P=(np.log(min_period), np.log(max_period)),
        )
    )

    gp = celerite.GP(kernel=kernel, mean=0.)
    gp.compute(t)
    return gp