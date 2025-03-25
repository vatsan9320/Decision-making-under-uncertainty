import sys
import os
from pyomo.environ import *

# Add the directory to the Python path
sys.path.append(os.path.abspath('C:/Users/luisa/OneDrive - Danmarks Tekniske Universitet/DTU/2year_2semester/Decision-making under uncertainty/AssignmentA/Assignment_Codes'))

# Import necessary modules
import data
import Plots
from PriceProcess import *
from WindProcess import *

# Load given data
data = data.get_fixed_data()

# Function to simulate wind and price processes
def simulate_wind_and_price(data):
    simulated_data = {'wind': {}, 'price': {}}

    # Generate wind power
    for t in range(data['num_timeslots']):
        if t == 0:
            simulated_data['wind'][t] = wind_model(data['wind_power_t_1'], data['wind_power_t_2'], data)
        elif t == 1:
            simulated_data['wind'][t] = wind_model(simulated_data['wind'][t-1], data['wind_power_t_1'], data)
        else:
            simulated_data['wind'][t] = wind_model(simulated_data['wind'][t-1], simulated_data['wind'][t-2], data)

    # Generate price process
    for t in range(data['num_timeslots']):
        if t == 0:
            simulated_data['price'][t] = price_model(data['price_t_1'], data['price_t_2'], simulated_data['wind'][t], data)
        elif t == 1:
            simulated_data['price'][t] = price_model(simulated_data['price'][t-1], data['price_t_1'], simulated_data['wind'][t], data)
        else:
            simulated_data['price'][t] = price_model(simulated_data['price'][t-1], simulated_data['price'][t-2], simulated_data['wind'][t], data)

    return simulated_data

# Function to solve the model for a given policy
def solve_model_with_policy(policy_constraints, simulated_data):
    # Create a model
    model = ConcreteModel()


    #------------------------------------------------------------------------------------------
    # Declare variables
    #  The electrolyzer can be ON or OFF at each hour t
    model.e = Var(range(data['num_timeslots']), within=Binary)
    # the amount of power to turn into hydrogen
    model.P2H = Var(range(data['num_timeslots']), bounds=(0, data['p2h_rate']))
    # the amount of hydrogen stored
    model.S = Var(range(data['num_timeslots']), bounds=(0, data['hydrogen_capacity']))
    # the amount of hydrogen turned back into power
    model.H2P = Var(range(data['num_timeslots']), bounds=(0, data['h2p_rate']))
    # power to take from the grid - change to be the power to take from the grid straigth to the industrial plant
    model.p = Var(range(data['num_timeslots']), bounds=(0, None))
    # binary if electrolizer can be turn on or not
    model.yon = Var(range(data['num_timeslots']), within=Binary)
    # binary if electrolizer can be turn off or not
    model.yoff = Var(range(data['num_timeslots']), within=Binary)
    #------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------
    # Objective function: Minimization of costs
    model.profit = Objective(
        expr=sum(model.e[t] * data['electrolyzer_cost'] + model.p[t] * simulated_data['price'][t] for t in range(data['num_timeslots'])),   
        sense=minimize)
    #------------------------------------------------------------------------------------------

    # Constraints
    #------------------------------------------------------------------------------------------
    # set initial status of the electrolyzer
    model.InitialStatus = ConstraintList()
    model.InitialStatus.add(model.e[0] == 0)   #TODO: can we assume this? We can

    # Constraint on available power always == demand
    model.Power = ConstraintList()
    for t in range(data['num_timeslots']):
        # power in the fabric = power from the grid + power from the wind + power from the hydrogen = demand
        # TODO: should be == ??? why is it infeasible? because we cannot stop wind
        model.Power.add(model.p[t] + simulated_data['wind'][t] + model.H2P[t]*data['conversion_h2p'] - model.P2H[t] >= data['demand_schedule'][t])
                        

    # there is a conversion rate from power to hydrogen
    model.P2H_Conversion = ConstraintList()
    for t in range(data['num_timeslots']):  
        # we can never convert at t=0 since we need to turn on the electrolyzer first
        model.P2H_Conversion.add(model.P2H[0] == 0)
        if t>0:
            # we convert if the electrolyzer is on and if we had surplus power from wind in t-1, meaning wind > demand
            # doesnt need to be equal because maybe we want to convert less than the surplus
            model.P2H_Conversion.add(model.P2H[t] <= model.e[t] * data['p2h_rate'])  
            #model.P2H_Conversion.add(model.P2H[t] <= model.e[t-1] * data['p2h_rate']) 

    # thre is a conversion rate from hydrogen to power
    model.H2P_Conversion = ConstraintList()
    for t in range(data['num_timeslots']):
        model.H2P_Conversion.add(model.H2P[t] <= model.S[t])

    #  When an amount of hydrogen is produced at t-1, it becomes stored and available 
    # in the tank from the next timeslot t, plus the amount of hydrogen already stored (t-1) 
    # and minus the amount of hydrogen turned back into power in the same period t
    model.Storage = ConstraintList()
    for t in range(data['num_timeslots']):
        if t == 0:
            model.Storage.add(model.S[0] == 0)
        else:
            model.Storage.add(model.S[t] == model.S[t-1] + model.P2H[t-1] * data['conversion_p2h']  - model.H2P[t])


    # if the electrolyzer is set to on we cannot turn it off and viceversa
    model.Electrolyzer = ConstraintList()
    for t in range(data['num_timeslots']):
        model.Electrolyzer.add(model.yon[t] + model.yoff[t] <= 1)

    # we can only turn it off if it is on
    model.Electrolyzer.add(model.e[t] >= model.yoff[t])

    # we can only turn it on if it is off
    model.Electrolyzer.add(model.e[t] <= 1- model.yon[t])

    # there is also relatioship between ele at time t and ele at time t-1 and yon and yoff at time t-1
    model.Electrolyzer.add(model.e[t] == model.e[t-1] + model.yon[t-1] - model.yoff[t-1])


    ################### APPLY POLICY CONSTRAINTS ####################

    # Apply the given policy constraints (important: this is fresh each time!)
    policy_constraints(model)

    # Solve the model
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=False)

    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        return value(model.profit)
    else:
        return None  # Return None if no optimal solution was found

# Define policies
def policy0(model):
    """ Policy 0: No constraints (baseline) """
    pass

def policy1(model):
    """ Policy 1: Always turn off the electrolyzer """
    model.policy1 = ConstraintList()
    for t in range(data['num_timeslots']):
        model.policy1.add(model.e[t] == 0)

def policy2(model):
    """ Policy 2: Only convert hydrogen to power if storage is above a threshold"""
    model.policy_h2p_threshold = ConstraintList()
    threshold = 0.3 * data['hydrogen_capacity'] 
    for t in range(data['num_timeslots']):
        model.policy_h2p_threshold.add(model.H2P[t] <= threshold)

def policy3(model):
    """ Policy: Convert excess wind power into hydrogen when wind > demand """
    model.policy_convert_excess_wind = ConstraintList()
    for t in range(data['num_timeslots']):
        model.policy_convert_excess_wind.add(model.P2H[t] <= max(0, simulated_data['wind'][t] - data['demand_schedule'][t]))


# Run multiple simulations
num_simulations = 10
results = {'policy0': [], 'policy1': [], 'policy2': [], 'policy3': []}

for _ in range(num_simulations):
    print(f"Simulation {_+1}/{num_simulations}")
    # Generate new simulated data
    simulated_data = simulate_wind_and_price(data)

    # Solve the model with each policy (fresh instance every time)
    result0 = solve_model_with_policy(policy0, simulated_data)
    result1 = solve_model_with_policy(policy1, simulated_data)
    result2 = solve_model_with_policy(policy2, simulated_data)
    result3 = solve_model_with_policy(policy3, simulated_data)

    # Store results
    results['policy0'].append(result0)
    results['policy1'].append(result1)
    results['policy2'].append(result2)
    results['policy3'].append(result3)

# Print final results
#print("Policy 1 Results:", results['policy1'])
#print("Policy 2 Results:", results['policy2'])

# print average results
print("Average Policy 0 Results:", sum(results['policy0']) / num_simulations)
print("Average Policy 1 Results:", sum(results['policy1']) / num_simulations)
print("Average Policy 2 Results:", sum(results['policy2']) / num_simulations)
print("Average Policy 3 Results:", sum(results['policy3']) / num_simulations)