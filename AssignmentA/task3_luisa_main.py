import numpy as np
from Assignment_Codes.WindProcess import wind_model
from Assignment_Codes.PriceProcess import price_model

# Define the feature vector for the state
def feature_vector(z, y):
    # z = (wind, price), y = (ele_status, storage)
    return np.array([z[0], z[1], y[0], y[1]])

# Linear value function approximation
def value_function(z, y, theta):
    x = feature_vector(z, y)
    return np.dot(theta, x)


def train_value_function(data, gamma=0.95, I=100, K=5):
    theta = np.random.randn(4)  # Linear model with 4 features

    for t in reversed(range(data['num_timeslots'])):
        X, Y_target = [], []

        for _ in range(I):
            wind_prev = data['wind_power_t_2']
            wind_curr = data['wind_power_t_1']
            # Simulate wind and price for the next time step
            wind_next = wind_model(wind_curr, wind_prev, data)
            price_prev = data['price_t_2']
            price_curr = data['price_t_1']
            # Simulate price based on wind and previous price
            price_next = price_model(price_curr, price_prev, wind_next, data)

            z_prev = (wind_curr, price_curr)
            z_t = (wind_next, price_next)

            # ---- Sample endogenous state y_t = (ele_status, storage) ----
            ele_status = np.random.choice([0, 1])
            storage = np.random.uniform(0, data['hydrogen_capacity'])
            y_t = (ele_status, storage)

            best_val = -np.inf

            for u in generate_candidate_controls(data, z_t, y_t):
                val_sum = 0
                for _ in range(K):
                    z_next = simulate_next_exogenous(z_t, z_prev, data)
                    y_next = simulate_next_endogenous(y_t, u, data)
                    val_sum += value_function(z_next, y_next, theta)

                expected_value = val_sum / K
                reward = reward_function(z_t, y_t, u, data)
                V_target = reward + gamma * expected_value

                best_val = max(best_val, V_target)

            # Store training example
            X.append(feature_vector(z_t, y_t))
            Y_target.append(best_val)

        # Fit theta using least squares
        X = np.array(X)
        Y_target = np.array(Y_target)
        theta = np.linalg.lstsq(X, Y_target, rcond=None)[0]

    return theta


def adp_policy(z_t, z_prev, y_t, theta, data, gamma=0.95, S=10):
    """
    Compute optimal decision u_t given current exogenous z_t, previous z_prev, and endogenous y_t.
    """
    best_action = None
    best_value = -np.inf

    for u in generate_candidate_controls(data, z_t, y_t):
        value_sum = 0
        for _ in range(S):
            z_next = simulate_next_exogenous(z_t, z_prev, data)
            y_next = simulate_next_endogenous(y_t, u, data)
            value_sum += value_function(z_next, y_next, theta)

        expected_value = (1 / S) * value_sum
        reward = reward_function(z_t, y_t, u, data)
        total_val = reward + gamma * expected_value

        if total_val > best_value:
            best_value = total_val
            best_action = u

    return best_action

def generate_candidate_controls(data, z, y):
    """
    Generate valid (P2H, H2P, p, yon, yoff) actions based on current state.
    """
    candidates = []
    ele_status, storage = y

    for yon in [0, 1]:
        for yoff in [0, 1]:

            # --- Enforce logical switching rules ---
            if yon and ele_status == 1:
                continue  # can't turn ON if already ON
            if yoff and ele_status == 0:
                continue  # can't turn OFF if already OFF
            if yon and yoff:
                continue  # can't turn ON and OFF at the same time

            for P2H in [0, data['p2h_rate'] if ele_status else 0]:  # only produce if ON
                for H2P in [0, min(data['h2p_rate'], storage)]:
                    for p in [0, 5, 10]:  # or range to max grid power
                        candidates.append((P2H, H2P, p, yon, yoff))

    return candidates


def simulate_next_exogenous(z_t, z_prev, data):
    """
    Simulate next exogenous state (wind, price) using your provided models.
    
    z_t    = (current_wind, current_price)
    z_prev = (previous_wind, previous_price)
    """
    current_wind, current_price = z_t
    previous_wind, previous_price = z_prev

    wind_next = wind_model(current_wind, previous_wind, data)
    price_next = price_model(current_price, previous_price, wind_next, data)

    return (wind_next, price_next)


def simulate_next_endogenous(y_t, u, data):
    ele_status, storage = y_t
    P2H, H2P, p, yon, yoff = u

    ele_next = ele_status + yon - yoff
    storage_next = storage + P2H * data['conversion_p2h'] - H2P
    storage_next = np.clip(storage_next, 0, data['hydrogen_capacity'])

    return (ele_next, storage_next)


def reward_function(z_t, y_t, u, data):
    wind, price = z_t
    ele_status, _ = y_t
    P2H, H2P, p, yon, yoff = u

    cost = ele_status * data['electrolyzer_cost'] + p * price
    # Add penalty or reward for unmet demand or stability if needed
    return -cost


# to run in the experiments
def adp_policy_wrapper(theta):
    def wrapper(data, wind, price, hydrogen_stock, ele, t, T):
        # Get states
        if t == 0:
            z_prev = (data['wind_power_t_1'], data['price_t_1'])
        elif t == 1:
            z_prev = (data['wind_power_t_1'], data['price_t_1'])
        else:
            z_prev = (wind[t - 2], price[t - 2])

        z_t = (wind[t - 1], price[t - 1])
        y_t = (ele[t], hydrogen_stock[t])

        return adp_policy(z_t, z_prev, y_t, theta, data)
    return wrapper
