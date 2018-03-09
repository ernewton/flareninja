import emcee
import autograd.numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle, median_absolute_deviation
from scipy.optimize import minimize
import glob
from tqdm import tqdm

import celerite
from celerite import terms

from gp import get_rotation_gp
from astropy.io import fits

# Do some aggressive sigma clipping
def sigma_clip(t, y, yerr, mask=None):

    if mask is not None:
        m = np.copy(mask)
    else:
        m = np.ones(len(t), dtype=bool)
    while True:
        mu = np.nanmean(y[m])
        sig = np.nanstd(y[m])
        m0 = y - mu < 3 * sig
        if np.all(m0 == m):
            break
        m = m0

    #t, y, yerr = t[m], y[m], yerr[m]
    return m


# First guess at the period
def ls_period(t,y, debug=False):
    
    fmin = max([2./(t[-1]-t[0]),0.02] )
    freq = np.linspace(fmin, 10.0, 5000)
    model = LombScargle(t, y)
    power = model.power(freq, method="fast", normalization="psd")
    power /= len(t)

    period = 1.0 / freq[np.argmax(power)]
    
    if debug:
        print(period)
        plt.figure()
        plt.plot(1.0 / freq, power, "k")
        plt.axvline(period)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
        
        plt.figure()
        plt.scatter(t % period, y, cmap='autumn', c=t)
        plt.show()
        
    return period


# functions for fitting    
def neg_log_like(params, y, gp, m):
    gp.set_parameter_vector(params)
    assert len(y) == len(m)
    return -gp.log_likelihood(y[m])

def grad_neg_log_like(params, y, gp, m):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y[m])[1]

# Iterate on GP model to do fitting with clipping
# uses simple minimization for ease
def iterate_gp(t, y, yerr, period, niter=10, mask=None):

    # get the kernel, which makes bounds choices based on period
    min_period = period * 0.8
    max_period = period / 0.8
 
    if mask is not None:
        m = np.copy(mask)
    else:
        m = np.ones_like(t, dtype=bool)

    gp = get_rotation_gp(t[m], y[m], yerr[m], period, min_period, max_period)
    gp.freeze_parameter("kernel:terms[2]:log_P")
    gp.freeze_parameter("kernel:terms[1]:log_sigma")
    
    gp.compute(t[m], yerr[m])
    gp.get_parameter_dict()

    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    # now iterate on the GP
    for i in range(niter):
        gp.compute(t[m], yerr[m])
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(y, gp, m))
        gp.set_parameter_vector(soln.x)
        initial_params = soln.x

        mu, var = gp.predict(y[m], t, return_var=True)
        sig = np.sqrt(var + yerr**2)

        m0 = y - mu < 1.3 * sig
        print(m0.sum(), m.sum())
        
        # break if nothing clipped, or keep going
        if np.all(m0 == m) or (np.abs(m0.sum()- m.sum()) < 3):
            break
        m = m0
    # all done

    # final fit
    fit_t, fit_y, fit_yerr = t[m], y[m], yerr[m]

    gp.thaw_parameter("kernel:terms[2]:log_P")
    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    gp.compute(fit_t, fit_y)
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(y, gp, m))
    gp.set_parameter_vector(soln.x)
    initial_params = soln.x

    print(gp.get_parameter_dict()) 
    mask = m

    return gp, soln, fit_y

 
# The priors: a multimodal prior on the period
# and default priors on everything else (which 
# does not include period, as bounds are None)
def additional_prior(params, p, logperiod):
    
    period = np.exp(logperiod)
    sigma = 0.2
    
    logperiod_half = np.log(period)*2.
    logperiod_twice = np.log(period)/2.
    gaussian_prior = (-1./2.)*((p - logperiod)/(sigma))**2.
    gaussian_prior_half = (-1./2.)*((p - logperiod_half)/(sigma))**2.
    gaussian_prior_twice = (-1./2.)*((p - logperiod_twice)/(sigma))**2.

    lp = 0.5*gaussian_prior + 0.25*gaussian_prior_half + 0.25*gaussian_prior_twice
    return lp

def log_prob(params, fit_y, gp, logperiod):

    gp.set_parameter_vector(params)
    p_current = gp.get_parameter_dict()['kernel:terms[2]:log_P']
    
    lp = gp.log_prior() + additional_prior(params, p_current, logperiod)
    if not np.isfinite(lp):
        return -np.inf

    return lp + gp.log_likelihood(fit_y)

    
def emcee_gp(gp, soln, fit_y,):

    ndim = len(soln.x)
    nwalkers = 32
    print(gp.get_parameter_dict()    )
    logperiod = gp.get_parameter_dict()['kernel:terms[2]:log_P']
    
    # set up initial positions
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
    lp = np.array( [log_prob(pos_i, fit_y, gp, logperiod) for pos_i in pos] )
    # and make sure none of them are NaNs
    m = ~np.isfinite(lp)
    while np.any(m):
        pos[m] = soln.x + 1e-5 * np.random.randn(m.sum(), ndim)
        lp[m] = np.array( [log_prob(pos_i, fit_y, gp, logperiod) for pos_i in pos[m]] )
        m = ~np.isfinite(lp)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(fit_y,gp, logperiod))
    ## note fit_y needs to be in [] if gp is not a param 
    ## in order to convince the mapping to get the right dimens
    
    # Run the burn-in
    pos, _, _ = sampler.run_mcmc(pos, 500)
    
    # Run the real chain
    #sampler.reset()
    #sampler.run_mcmc(pos, 1000);    

    # Get the posterior
    mle = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    gp.set_parameter_vector(mle)

    return gp, sampler


               
def lc_plot(ax, t_orig, y_orig, mu, std, xlim=None):
    color = "#ff7f0e"
    ax.fill_between(t_orig, mu+std*3, mu-std*3, alpha=0.7, color=color, zorder=0)
    ax.plot(t_orig, y_orig, '.', zorder=1)
    if xlim is None:
        xlim = [t_orig[0], t_orig[-1]]
    ax.set_xlim(xlim)
    use, = np.where( (t_orig > xlim[0]) & (t_orig < xlim[1]) )
    ylow = 1.1*np.nanmin(y_orig[use])
    yhigh = 1.1*np.nanmax(y_orig[use])
    ax.set_ylim(ylow,yhigh)
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Relative Brighness', fontsize=14)

def post_plot(ax, dist):
    ax.hist(dist, 50, histtype="step")
    print ax.get_xlim()
    ylim = ax.get_ylim()
    low, mid, high = np.percentile( dist, [16, 50, 84])
    plt.plot([mid,mid],ylim, c='indianred')
    print low, mid, high
    for lh in (low, high):
        plt.plot([lh,lh],ylim, ':', c='indianred')
    ax.set_xlabel('Rotation Period (days)', fontsize=14)
    ax.set_ylabel('Posterior Probability', fontsize=14)



