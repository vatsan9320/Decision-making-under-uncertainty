# only check if the decisions are feasible

def check_feasibility(data, y_on, y_off, P2H, H2P, p, wind, demand, current_hydrogen, current_ele):
    """
    Check whether the policy's here-and-now decisions are feasible/meaningful.
    Correct any infeasible values.
    data, y_on[e, t], y_off[e, t], P2H[e, t], H2P[e, t], p[e, t], wind[t],
                                                demand[t], hydrogen_stock[e, t], ele[e, t], ele[e, t-1], 
                                                y_on[e, t-1], y_off[e, t-1]
    """
    # Ensure P2H and H2P are within limits
    # Ensure H2P does not exceed available hydrogen
    H2P = min(max(H2P, 0), data['h2p_rate'], current_hydrogen)
    P2H = min(max(P2H, 0), data['p2h_rate'])
    

    # Ensure power production is non-negative
    p = max(p, 0)
    
    # Ensure demand is met
    total_supply = p + wind + H2P * data['conversion_h2p'] - P2H
    if total_supply < demand:
        print('total_supply < demand')
        print('p:', p, 'wind:', wind, 'H2P:', H2P, 'P2H:', P2H, 'total_supply', total_supply, 'demand:', demand)
        return False, y_on, y_off, P2H, H2P, p  # Not feasible, return unchanged values

    # Ensure the unit is either on or off, but not both
    if y_on + y_off > 1:
        print('y_on + y_off > 1')
        print('y_on:', y_on, 'y_off:', y_off)
        return False, y_on, y_off, P2H, H2P, p

    # Ensure energy storage consistency
    #if ele < y_off or ele > 1 - y_on:
    #    print('ele < y_off or ele > 1 - y_on')
    #    print('ele:', ele, 'y_off:', y_off, 'y_on:', y_on)
    #    return False, y_on, y_off, P2H, H2P, p
    #
    #if ele != ele_previous + y_on_previous - y_off_previous:
    #    print('ele != ele_previous + y_on_previous - y_off_previous')
    #    print('ele:', ele, 'ele_previous:', ele_previous, 'y_on_previous:', y_on_previous, 'y_off_previous:', y_off_previous)
    #    return False, y_on, y_off, P2H, H2P, p

    return True, y_on, y_off, P2H, H2P, p
