import k2plr
kclient = k2plr.API()
from appaloosa import detrend

# these are nearby M dwarfs with short cadence Kepler data
#kid = [9726699,8451881,201885041]
kid = [9726699,8451881]
          
for k in kid:
    if k>100000000:
        star = kclient.k2_star(k)
    else:
        star = kclient.star(k)
    lcs = star.get_light_curves(short_cadence=True, fetch=True)
    name = str(kid)+'.pkl'
    time, flux, ferr, quality = [], [], [], []
    for lc in lcs[0:1]:
        with lc.open() as f:
            if f[0].header['OBSMODE'] == 'short cadence':
                print star, lc
                hdu_data = f[1].data
                time.append(hdu_data["time"])
                flux.append(hdu_data["sap_flux"])
                ferr.append(hdu_data["sap_flux_err"])
                quality.append(hdu_data["sap_quality"])


         
box1 = detrend.MultiBoxcar(time_i[fin], flux_i[fin], err_i[fin], kernel=2.0, numpass=2)
sin1, per = detrend.FitSin(time_i[fin], box1, err_i[fin], maxnum=5, maxper=(max(time_i)-min(time_i)        