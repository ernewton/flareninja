import k2plr
kclient = k2plr.API()

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
    for lc in lcs:
        with lc.open() as f:
            if f[0].header['OBSMODE'] == 'short cadence':
                print star, lc
                hdu_data = f[1].data
                time.append(hdu_data["time"])
                flux.append(hdu_data["sap_flux"])
                ferr.append(hdu_data["sap_flux_err"])
                quality.append(hdu_data["sap_quality"])

            
