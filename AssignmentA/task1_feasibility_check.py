# only check if the decisions are feasible

def check_feasibility(data, y_on, y_off, P2H, H2P, p, wind, hydrogen_stock, ele, t):
    """
    Check whether the policy's here-and-now decisions are feasible/meaningful.
    Correct any infeasible values.
    """

    demand = data['demand_schedule'][t]
    hydrogen_stock = hydrogen_stock[t]
    wind = wind[t]


    # Ensure P2H and H2P are within limits
    # Ensure H2P does not exceed available hydrogen
    H2P = min(max(H2P, 0), data['h2p_rate'], hydrogen_stock)
    P2H = min(max(P2H, 0), data['p2h_rate'])
    

    # Ensure power production is non-negative
    p = max(p, 0)
    
    # Ensure demand is met
    total_supply = p + wind + H2P * data['conversion_h2p'] - P2H 
    if total_supply  + 1e-6 < demand :  # Add epsilon to avoid tight infeasibility
        print('total_supply < demand')
        print('p:', p, 'wind:', wind, 'H2P:', H2P, 'P2H:', P2H, 'total_supply', total_supply, 'demand:', demand)
        return False, y_on, y_off, P2H, H2P, p  # Not feasible, return unchanged values

    # Ensure the unit is either on or off, but not both
    if y_on + y_off > 1:
        print('y_on + y_off > 1')
        print('y_on:', y_on, 'y_off:', y_off)
        return False, y_on, y_off, P2H, H2P, p

    return True, y_on, y_off, P2H, H2P, p
