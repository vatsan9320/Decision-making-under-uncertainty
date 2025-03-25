# only check if the decisions are feasible

def correct_decision(data, y_on, y_off, P2H, H2P, p, wind, demand, current_hydrogen, current_ele):
    #use case
    # we have decided to convert 

    # Ensure demand is met
    total_supply = p + wind + H2P * data['conversion_h2p'] - P2H
    if total_supply < demand:
        # take from the 

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
