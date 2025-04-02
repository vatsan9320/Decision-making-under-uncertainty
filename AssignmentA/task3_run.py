# Import necessary modules
import matplotlib.pyplot as plt
from Assignment_Codes.data import get_fixed_data
from task3_luisa import *



data = get_fixed_data()
# Initial exogenous states
z_prev = (data['wind_power_t_2'], data['price_t_2'])  # wind_t-2, price_t-2
z_t = (data['wind_power_t_1'], data['price_t_1'])     # wind_t-1, price_t-1

# Initial endogenous state
y_t = (1, 10)  # electrolyzer ON, 10 units of storage

theta = train_value_function(data, gamma=0.95, I=50, K=5)
print("Trained theta:", theta)



T = data['num_timeslots']  # Number of time steps
trajectory = []

for t in range(T):
    # Make decision
    u = adp_policy(z_t, z_prev, y_t, theta, data)

    # Store the state and decision
    trajectory.append({
        't': t,
        'z_prev': z_prev,
        'z_t': z_t,
        'y_t': y_t,
        'u': u
    })

    # Transition to next step
    z_next = simulate_next_exogenous(z_t, z_prev, data)
    y_next = simulate_next_endogenous(y_t, u, data)

    # Shift states forward
    z_prev = z_t
    z_t = z_next
    y_t = y_next

#for entry in trajectory:
#    print(f"Time {entry['t']}:")
#    print(f"  Wind: {entry['z_t'][0]:.2f}, Price: {entry['z_t'][1]:.2f}")
#    print(f"  Electrolyzer: {entry['y_t'][0]}, Storage: {entry['y_t'][1]:.2f}")
#    print(f"  Action (P2H, H2P, p, yon, yoff): {entry['u']}")
#    print("-" * 40)



# Extract time series
time = [entry['t'] for entry in trajectory]
wind = [entry['z_t'][0] for entry in trajectory]
price = [entry['z_t'][1] for entry in trajectory]
ele_status = [entry['y_t'][0] for entry in trajectory]
storage = [entry['y_t'][1] for entry in trajectory]
P2H = [entry['u'][0] for entry in trajectory]
H2P = [entry['u'][1] for entry in trajectory]
grid_power = [entry['u'][2] for entry in trajectory]

# Optional: yon/yoff actions
yon = [entry['u'][3] for entry in trajectory]
yoff = [entry['u'][4] for entry in trajectory]

# ---- PLOT SECTION ----
plt.figure(figsize=(14, 8))

# Wind and Price
plt.subplot(2, 2, 1)
plt.plot(time, wind, label='Wind Power')
plt.plot(time, price, label='Price')
plt.title("Wind and Price")
plt.xlabel("Time")
plt.legend()

# Electrolyzer & Storage
plt.subplot(2, 2, 2)
plt.plot(time, ele_status, label='Electrolyzer Status (0/1)')
plt.plot(time, storage, label='Hydrogen Storage')
plt.title("Electrolyzer & Storage")
plt.xlabel("Time")
plt.legend()

# Power Flows
plt.subplot(2, 2, 3)
plt.plot(time, P2H, label='P2H')
plt.plot(time, H2P, label='H2P')
plt.plot(time, grid_power, label='Grid Power')
plt.title("Power Flows")
plt.xlabel("Time")
plt.legend()

# On/Off Actions
plt.subplot(2, 2, 4)
plt.plot(time, yon, label='yon (Turn On)')
plt.plot(time, yoff, label='yoff (Turn Off)')
plt.title("Electrolyzer Switching")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.show()
