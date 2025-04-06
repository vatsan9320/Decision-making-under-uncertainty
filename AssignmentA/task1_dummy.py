def make_dummy_decision(data, wind, price, hydrogen, ele, t, T):
    """
    Dummy policy:
    - Always turns off electrolyzer
    - Never uses P2H
    - Never uses H2P
    - Always uses the grid to meet all demand
    """
    # Always turn off electrolyzer
    y_on = 0
    y_off = 1 if ele == 1 else 0

    P2H = 0
    H2P = 0
    p = data['demand_schedule'][t] - wind[t]
    p = max(p, 0) + 1e-6  # Add epsilon to avoid tight infeasibility

    return y_on, y_off, P2H, H2P, p
