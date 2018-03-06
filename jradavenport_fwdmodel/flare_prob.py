'''
Methods here build the master flare-rate model. This will be used as:

Given the 1 parameter that tunes the flare rate (FFD amplitude),
 what is the cumulative distribution of the relative flux?

I want this relationship be to be analytic, so it will be very fast and
applicable to a zillion sparsely sampled light curves with ease.

'''


import numpy as np
import aflare # flare morphology (Davenport et al. 2014)
import matplotlib.pyplot as plt


def _randomp(num, slope=-2, mini=0.1, maxi=10.):
    # eqn based on this:
    # http://mathworld.wolfram.com/RandomNumber.html
    p = ((maxi**slope - mini**slope) * np.random.random(num) + mini**slope)**(1. / slope)

    return p


def SuperLC(ffd_alpha=1.0, ffd_beta=-2.0, dur=1.0, repeat=0, mag=False,
            display=False, ffd_min=0, ffd_max=5, dt = 0.1 / 24. / 60.):
    '''
    generate a super-sampled light curve of flares for the duration

    FFD fit must be in log(cumulative # / day) _versus_ log(Equiv Dur)

    ffd_beta = slope, should be negative
    ffd_alpha = intercept, sets the overall rate (log #, e.g. 1.0, -2.0)
    dur = duration in years
    repeat: how many times to replicate the data, saves computation time

    # log ED limits: ffd_min, ffd_max
    '''

 
    time = np.arange(0, dur * 365., dt)
    print('Making ' + str(len(time)) + ' epochs')

    # to calc the # flares, evaluate the FFD @ the minimum energy,
    # then multiply by the total duration in days
    Nflares = int(np.floor(np.power(10.0, ffd_min*ffd_beta + ffd_alpha)) * (dur * 365.0))
    print('Simulating ' + str(Nflares) + ' flares')

    f_energies = _randomp(Nflares, slope=ffd_beta, mini=10.0**ffd_min, maxi=10.0**ffd_max)

    # make some REALLY BAD assumptions from event energy to FWHM and Amplitude
    fwhm = (10**((np.log10(f_energies) + 0.5) / 1.5)) / 24. / 60.
    ampl = f_energies/2.


    # put flares at random places throughout light curve
    t_peak = np.random.random(Nflares) * (dur * 365)

    flux = np.zeros_like(time) # a blank array of fluxes
    for k in range(0, int(Nflares)):
        flux = flux + aflare.aflare1(time, t_peak[k], fwhm[k], ampl[k])
    #

    if repeat > 0:
        tout = time
        for i in range(1,int(repeat)):
            tout = np.hstack((tout, time + max(tout)))
        time = tout
        flux = np.tile(flux, int(repeat))

    if mag is True:
        flux = -2.5 * np.log10(flux + 1.0)

    if display is True:
        plt.figure()
        plt.plot(time, flux)
        plt.xlabel('Time (days)')
        plt.ylabel('Flux')
        plt.show()

    return time, flux


# def setalpha(gi, logage):
#     '''
#     This encodes the really tough stuff from my work, which isn't done yet...
#         how to set the flare rate amplitude as a function of a star's mass and age!
#
#     Importantly: This "surface" is currently unknown!
#
#     use g-i color instead of mass or temperature. assume main sequence, of course.
#
#     for starting, use a flat surface polynomial over color and log(age) of the form
#     alpha = k*color + j*logage + b
#     '''
#
#     # WARNING: these numbers are totally made up
#     k = -0.1 # change in alpha as function of color
#     j = -0.5 # change in alpha as function of log(age)
#     b = 1.0 # baseline value of alpha
#
#     alpha = k*gi + j*logage + b
#
#     return alpha


def CDist(num=4):
    '''
    For a range of FFD amplitudes (holding the slope fixed)
    - Generate their super-sampled LC's
    - Convert each LC to cumulative fractional flux distribution versus fraction of time
    - Do a 3-D (surface) fit, that parametrizes the cumulative flux distribution versus the 1 free FFD parameter
    - Return the coefficients for this surface fit to be used in the toy model

    '''


    ffd_intercept = np.logspace(-3, 0, num=num)


    plt.figure()

    for k in range(0, num):
        print(k)

        lc = SuperLC(ffd_alpha=ffd_intercept[k], dur=0.1)
        lc.sort()

        frac = np.arange(0, len(lc))/ float(len(lc))
        plt.plot(lc, frac)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Delta Flux Value')
    plt.ylabel('Fraction of Time')
    plt.show()

    return


if __name__ == "__main__":
    # CDist()
    SuperLC(dur=0.1, repeat=1, display=True, mag=False)
