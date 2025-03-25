import sys
import os
from pyomo.environ import *
import numpy as np


# Import necessary modules
from Assignment_Codes.data import get_fixed_data
from Assignment_Codes.Plots import plot_results
from Assignment_Codes.PriceProcess import *
from Assignment_Codes.WindProcess import *
from task1_feasibility_check import check_feasibility
from task1_policy import make_decision
from task1_nextstate import nextstate


# Load given data
data = get_fixed_data()

n_timeslots = data['num_timeslots']

# Function to simulate wind and price processes
def simulate_wind_and_price(data):
    simulated_data = {'wind': {}, 'price': {}}

    # Generate wind power
    for t in range(n_timeslots):
        if t == 0:
            simulated_data['wind'][t] = wind_model(data['wind_power_t_1'], data['wind_power_t_2'], data)
        elif t == 1:
            simulated_data['wind'][t] = wind_model(simulated_data['wind'][t-1], data['wind_power_t_1'], data)
        else:
            simulated_data['wind'][t] = wind_model(simulated_data['wind'][t-1], simulated_data['wind'][t-2], data)

    # Generate price process
    for t in range(n_timeslots):
        if t == 0:
            simulated_data['price'][t] = price_model(data['price_t_1'], data['price_t_2'], simulated_data['wind'][t], data)
        elif t == 1:
            simulated_data['price'][t] = price_model(simulated_data['price'][t-1], data['price_t_1'], simulated_data['wind'][t], data)
        else:
            simulated_data['price'][t] = price_model(simulated_data['price'][t-1], simulated_data['price'][t-2], simulated_data['wind'][t], data)

    return simulated_data


# Initialize decision variables 
y_on = {}
y_off = {}
P2H = {}
H2P = {}
p = {}

# we also need to initilize state variables no???
hydrogen_stock = {}
ele = {}	

n_experiments = 100
n_timeslots = data['num_timeslots']
# Intialize policy experiment policy cost and policy cost
policy_cost = np.full(n_timeslots, 99999999)
policy_cost_at_experiment = np.full(n_experiments, 99999999)

# for each experiment
for e in range(n_experiments):
    # Simulate wind and price processes
    simulated_data = simulate_wind_and_price(data)
    wind = simulated_data['wind']
    price = simulated_data['price']
    demand = data['demand_schedule']
    policy_cost = 0
    
    # Set initial hydrogen stock
    hydrogen_stock[e, -1] = 0
    ele[e, -1] = 0

    hydrogen_stock[e, 0] = 0
    ele[e, 0] = 0

    y_on[e, -1] = 0
    y_off[e, -1] = 0
    P2H[e, -1] = 0
    H2P[e, -1] = 0
    p[e, -1] = 0

    # for each timeslot of the horizon
    for t in range(n_timeslots):


        # Call policy to make a decision for here and now
        y_on[e, t], y_off[e, t], P2H[e, t], H2P[e, t], p[e, t] = make_decision(wind[t], price[t], hydrogen_stock[e,t], ele[e,t], demand[t]) 
        # it is t, because the states are always dependent on the previous state + decisions taken there, so when we are in t, we have already
        # the the state of t, an we can take decionsions in t that will impact the state in t+1
        

        # check whether the policy's here-and-now decisions are feasible/meaningful
        successful,  y_on[e, t], y_off[e, t], P2H[e, t], H2P[e, t], p[e, t] = check_feasibility(
                                    data, y_on[e, t], y_off[e, t], P2H[e, t], H2P[e, t], p[e, t], wind[t], demand[t], hydrogen_stock[e,t], ele[e,t]
                                                                                                )



        if not successful:
            print(f'Policy is not feasible at experiment {e}, time {t}')
            # print varaibles values
            print('y_on: ', y_on[e, t])
            print('y_off: ', y_off[e, t])
            print('P2H: ', P2H[e, t])
            print('H2P: ', H2P[e, t])
            print('p: ', p[e, t])
            break

        # Update policy cost
        policy_cost = policy_cost + (p[e, t] * price[t] + ele[e,t] * data['electrolyzer_cost'])

        # Update hydrogen stock
        hydrogen_stock[e,t+1], ele[e,t+1] = nextstate(
                                data, hydrogen_stock[e,t], ele[e,t], P2H[e,t], H2P[e,t], y_on[e,t], y_off[e,t]
                                    	                )
        


    # Update policy cost at experiment
    #TODO: if it is not feasible, we should not update the policy cost
    policy_cost_at_experiment[e] = policy_cost
    

# average of policy cost above all experiments
policy_cost = np.mean(policy_cost_at_experiment)

# print results
print('Policy cost: ', policy_cost)
print('Policy cost at experiment: ', policy_cost_at_experiment)



#successful = check_feasibility(
#    y_on[e, t],
#    y_on[e, t-1], #previous
#    y_off[e, t],
#    y_off[e, t-1], # previous
#    P2H[e, t],
#    H2P[e, t],
#    p[e, t],
#    wind[t],
#    #price,
#    ele[e, t],
#    ele[e, t-1], #previous
#    demand, #demand
#    hydrogen_stock[e,t] #S
#    #,hydrogen_stock[e,t-1] #previous
#)
