from fitstar import *
import corner
import pickle
import k2plr
from statsmodels import robust
  

def one_star(t_orig, y_orig, yerr_orig, name, mask=None):
    
    if mask is None:
        mask = np.isfinite(y_orig) & np.isfinite(yerr_orig)
    else:
        mask = mask & np.isfinite(y_orig) & np.isfinite(yerr_orig)
    
    f = lambda x: np.ascontiguousarray(x, dtype=np.float64)
    t, y, yerr = map(f, [t_orig[mask], y_orig[mask], yerr_orig[mask]])
    
    ###############
    # Do the fitting
    ###############

    # First get the period from a Lomb-Scargle periodogram
    m = sigma_clip(t,y,yerr) ## do sigma clipping
    period = ls_period(t[m],y[m], debug=False) ## get peak of periodogram
    print("LS period",period)

    # Next use simple minimization in order to iterate on
    # GP fits, clipping flares each time
    gp, fit_y = iterate_gp(t, y, yerr, period, mask=m)
    
    # Next use emcee to fit for the GP a final time
    gp, sampler = emcee_gp(gp, fit_y)

    ## do I actually recompute? I think what's currently here is right
    #gp.compute(t,yerr) 
    mu, var = gp.predict(fit_y, t_orig[mask], return_var=True)
    std = np.sqrt(yerr_orig[mask]**2 + var)
    ndim = len(sampler.flatchain[0, :])

    ###############
    # Make a dictionary of the results
    ###############

    # param best values marginalized over everything else
    varnames = gp.get_parameter_dict().keys()
    samples = sampler.chain[:, :, :].reshape((-1, ndim)) 
    best = map(lambda v: [v[1], v[2]-v[1], v[1]-v[0]], \
                       zip(*np.percentile(samples, [16, 50, 84], axis=0))) 
                       ## arranged: [50th, upper, lower]
    mydict = {}
    labels = [None]*ndim
    for i in range(len(varnames)):
        vv = varnames[i][16:]
        if vv == 'mix_par':
            vv = 'mix'
        else:
            vv = vv.replace('log_','log(')+')'
        mydict[vv] = best[i]
        labels[i] = vv

    pickle.dump(mydict, open(name+'.pkl','wb'))

    
    ###############
    # Plot the lightcurves
    ###############
    
    fig = plt.figure(figsize=[11,3])
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2, )
    ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1, )

    # set the x limits based on the best fit period
    n, = np.where(np.array(labels) == 'log(P)')
    pdist = np.exp(sampler.flatchain[:, n])
    pbest = np.median(pdist)
    xlim = [t[0], t[-1]]
    if (xlim[1]-xlim[0]) > 10*pbest:
        xlim = [t[0], t[0]+10*pbest]
        
    # actually make the plots
    lc_plot(ax1, t_orig[mask], y_orig[mask], mu, std, xlim=xlim)
    post_plot(ax2, pdist)

    fig.tight_layout()
    plt.savefig(name+'_per.pdf')

        
    ###############
    # Make a corner plot!
    ###############

    plt.figure()
    corner.corner(samples, labels=labels, 
                  quantiles=[.16, .50, .84], show_titles=True)   
    plt.savefig(name+'_corner.pdf')
    
        
    
def kic_test():

    ###############
    # Read in the data
    ###############

    f = fits.open('../data/kplr009726699-2009350155506_llc.fits') 
    hdu_data = f[1].data

    t = hdu_data["time"]
    y = hdu_data["sap_flux"]/np.nanmedian(hdu_data["sap_flux"])-1.
    yerr = hdu_data["sap_flux_err"]/np.nanmedian(hdu_data["sap_flux"])
    mask = hdu_data["sap_quality"] == 0
    
    #ninc = 5000
    #rand = np.random.randint(0,len(t)-ninc)
    #t = t[rand:rand+ninc]
    #y = y[rand:rand+ninc]
    #yerr = yerr[rand:rand+ninc]

    name = f[0].header['OBJECT'].replace(' ','')
    
    one_star(t, y, yerr, name, mask=mask)

    
def kic_llc(k, quarter=9): ## 9 or 3!
    kclient = k2plr.API()
    if k>100000000: 
        star = kclient.k2_star(k) ## K2
    else:
        star = kclient.star(k) # Kepler

    lcs = star.get_light_curves(short_cadence=False)

    quarters = np.zeros_like(lcs, dtype=int)
    for i, lc in enumerate(lcs):
        hdu_list = lc.open()
        quarters[i] = hdu_list[0].header['QUARTER']
        hdu_list.close()

    qq, = np.where(quarters == 9)
    if len(qq) == 0:
        print("No Q9", k)
        return False
    if len(qq) > 1:
        print("Two or more", k)
        return False
    
    lc = lcs[qq[0]]
    with lc.open() as f:
        hdu_data = f[1].data
        time = hdu_data["time"]
        flux = hdu_data["sap_flux"]/np.nanmedian(hdu_data["sap_flux"])-1.
        ferr = hdu_data["sap_flux_err"]/np.nanmedian(hdu_data["sap_flux"])
        mask = hdu_data["sap_quality"] == 0
        name = f[0].header['OBJECT'].replace(' ','')
       
        one_star(time, flux, ferr, name, mask)
        
        return True
   
 
def test(num):

    name = format('{0:04d}'.format(num))    
    ###############
    # Read in the data
    ###############

    t, y = np.genfromtxt('/Volumes/Mimas/full_dataset/final/lightcurve_'+name+'.txt',
                        unpack=True)
    y = y/np.median(y) - 1.
    
    ## need to estimate the error
    ## should multiple by 1.48, and then divide by sqrt(2) so pretty much good as is
    err_est = robust.mad(np.diff(y)) 
    yerr = np.ones_like(y)*err_est
    one_star(t, y, yerr, name)

    ## for now get approx. one quarter
    ninc = 5000
    rand = 0 #np.random.randint(0,len(t)-ninc)
    t = t[rand:rand+ninc]
    y = y[rand:rand+ninc]
    yerr = yerr[rand:rand+ninc]
        


