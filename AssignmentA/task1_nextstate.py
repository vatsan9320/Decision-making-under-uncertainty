def nextstate(data, hydrogen_stock_current, eletrolizer_status_current, P2H_current, H2P_current, y_on_current, y_off_current):
    next_hydrogen_stock = hydrogen_stock_current + P2H_current*data['conversion_h2p'] - H2P_current
    next_eletrolizer_status = eletrolizer_status_current + y_on_current - y_off_current
    return next_hydrogen_stock, next_eletrolizer_status
