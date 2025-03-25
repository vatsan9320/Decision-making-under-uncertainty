def make_decision(wind, price, hydrogen, ele, demand):
    # my policy will be that the electrolyzer is always off
    y_on = 0
    y_off = 0
    P2H = 0
    H2P = 0
    p = 2
    #if demand > wind:
    #    p = demand - wind
    #else:
    #    p = 0

    return y_on, y_off, P2H, H2P, p