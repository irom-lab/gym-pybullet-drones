def wind_function(time: float, gust_peak_time: float, gust_type: int) -> float:
    """
    This function returns a scalar gust value depending on the time and gust type.
    Based on: Cole, Kenan, and Adam Wickenheiser. "Spatio-Temporal Wind Modeling for UAV Simulations." arXiv e-prints (2019): arXiv-1905.
    
    param@ time: current time [s]
    param@ gust_peak_time: time of gust peak [s]
    param@ gust_type: integer to select gust options.
        gust_type == 0 -> semi-square impulse
        gust_type == 1 -> custom (change values in the function as you see fit)
        ...            -> add additional types as desired

    return@ float representing a 'gust scalar', to be added to a nominal wind [m/s]
    """
    if gust_type == 0: #Semi-Square Pulse
        tg1 = 0.1 #rising time
        tg2 = 0.1 #falling time
        th = 1  #holding time
        g1 = 1  #peak gust value
        g3 = 6 #change 'curvature'
        g4 = .1 #change magnitude of dip before gust
        g5 = .1 #change magnitude of dip after gust
    elif gust_type == 1: #Custom
        tg1 = 1 #rising time
        tg2 = 1 #falling time
        th = 1  #holding time
        g1 = 1  #peak gust value
        g3 = .3 #change 'curvature' of rising and falling sections
        g4 = 1 #change magnitude of dip before gust
        g5 = 2 #change magnitude of dip after gust
    else:
        print('No known gust_type specified.')
        return 0

    g2r = 2*g3/tg1
    g2f = 2*g3/tg2
    
    t = time - gust_peak_time
    
    if t<0.5*tg1: #Rising portion
        return g1*(1-(g2r*t-g3)**2)*np.exp(-(g2r*t-g3)**2/g4)
    elif t<0.5*tg1+th: #Holding portion
        return g1
    else: #Falling portion
        tstar = t + 0.5*(tg2 - tg1)-th
        return g1*(1-(g2f*tstar-g3)**2)*np.exp(-(g2f*tstar-g3)**2/g5)