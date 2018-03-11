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
def iterate_gp(t, y, yerr, period, niter=2, mask=None):

    # get the kernel, which makes bounds choices based on period
    min_period = period * 0.8
    max_period = period / 0.8
 
    if mask is not None:
        m = np.copy(mask)
    else:
        m = np.ones_like(t, dtype=bool)

    gp = get_rotation_gp(t[m], y[m], yerr[m], period, min_period, max_period)
    gp.compute(t[m], yerr[m])
    print(len(t[m]), np.sum(y[m]), np.sum(yerr[m]))
    print(gp.log_likelihood(y[m]))
    gp.freeze_parameter("kernel:terms[2]:log_P")
    
    
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
    # recompute (in case ran through all iters)
    gp.compute(fit_t, fit_yerr) 

    gp.thaw_parameter("kernel:terms[2]:log_P")
    return gp, fit_y

 
# The priors: a multimodal prior on the period
# and default priors on everything else (which 
# does not include period, as bounds are None)
def additional_prior(params, p, logperiod):
    
    sigma = 0.2 ## approx. 20% error
    
    logperiod_half = logperiod + np.log(0.5)
    logperiod_twice = logperiod + np.log(2.)
    gaussian_prior = (-1./2.)*((p - logperiod)/(sigma))**2.
    gaussian_prior_half = (-1./2.)*((p - logperiod_half)/(sigma))**2.
    gaussian_prior_twice = (-1./2.)*((p - logperiod_twice)/(sigma))**2.
 
    # but don't let it go too far
    if (np.abs(p-logperiod)>0.4) & (np.abs(p-logperiod_half)>0.4) & (np.abs(p-logperiod_twice)>0.4):
        return -np.inf
 
    lp = 0.5*gaussian_prior + 0.25*gaussian_prior_half + 0.25*gaussian_prior_twice
    return lp


def log_prob(params, fit_y, gp, logperiod):

    gp.set_parameter_vector(params)
    p_current = gp.get_parameter_dict()['kernel:terms[2]:log_P']
    
    lp = gp.log_prior() + additional_prior(params, p_current, logperiod)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gp.log_likelihood(fit_y)

    
def emcee_gp(gp, fit_y, to_convergence=False):
    np.random.seed(82)
    initial_params = gp.get_parameter_vector()
    logperiod = gp.get_parameter_dict()['kernel:terms[2]:log_P']
    ndim = len(initial_params)
    nwalkers = 50
    
    # set up initial positions
    pos = initial_params + 1e-1 * np.random.randn(nwalkers, ndim)

    # but for period start some at the harmonics
    tmp = [name == 'kernel:terms[2]:log_P' for name in gp.get_parameter_names()]
    perloc = np.where(tmp)[0][0] ## grab location of the entry for period
    for i in range(nwalkers):
        myrand = np.random.uniform()
        if myrand < 0.25: ## 25% at half the period
            pos[i][perloc] = logperiod + np.log(0.5) + 1e-1 * np.random.randn()
        elif myrand < 0.5: ## 25% at twice the period
            pos[i][perloc] = logperiod + np.log(2) + 1e-1 * np.random.randn()
      
    # and make sure none of the walkers start out of range
    lp = np.array( [log_prob(pos_i, fit_y, gp, logperiod) for pos_i in pos] )
    m = ~np.isfinite(lp)
    while np.any(m):
        pos[m] = initial_params + 1e-2 * np.random.randn(m.sum(), ndim)
        lp[m] = np.array( [log_prob(pos_i, fit_y, gp, logperiod) for pos_i in pos[m]] )
        m = ~np.isfinite(lp)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[fit_y,gp, logperiod])
    ## note fit_y needs to be in [] if gp is not a param 
    ## in order to convince the mapping to get the right dimens
    
    # Run the burn-in
    print("Running burn-in")
    pos, _, _ = sampler.run_mcmc(pos, 500)
    
    # Run the real chain
    sampler.reset()
    print("Running chain")
    sampler.run_mcmc(pos, 2000);    
    if to_convergence: ## this is culled from dfm/rotate
        print("Running to convergence")
        old_tau = np.inf
        autocorr = []
        converged = False
        mciter = 500
        totiter = 2500
        for iteration in range(10):
            pos, _, _ = sampler.run_mcmc(pos, mciter) # emcee3 takes thin_by=10)
            totiter = totiter + mciter

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            # this was my workaround to 
            tau = sampler.get_autocorr_time(c=0., high=mciter/10.)  
            autocorr.append(np.mean(tau))
            print(autocorr[-1])

            # Check convergence
            converged = np.all(tau * 100 < totiter)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.1)
            if converged:
                break
            old_tau = tau

        if converged:
            print("converged")
        else:
            print("not converged")       
        
    # Get the posterior, use max likelihood because median could be anywhere
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
    ylim = ax.get_ylim()
    low, mid, high = np.percentile( dist, [16, 50, 84])
    plt.plot([mid,mid],ylim, c='indianred')
    for lh in (low, high):
        plt.plot([lh,lh],ylim, ':', c='indianred')
    ax.set_xlabel('Rotation Period (days)', fontsize=14)
    ax.set_ylabel('Posterior Probability', fontsize=14)



