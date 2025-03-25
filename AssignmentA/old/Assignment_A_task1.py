import sys
import os

# Add the directory to the Python path
sys.path.append(os.path.abspath('C:/Users/luisa/OneDrive - Danmarks Tekniske Universitet/DTU/2year_2semester/Decision-making under uncertainty/AssignmentA/Assignment_Codes'))
#sys.path.append(os.path.abspath('../Decision-making under uncertainty/AssignmentA/Assignment_Codes'))

# import specific modules from folder AssignmentA/Assignment_Codes
import data
import Plots
# given data
data = data.get_fixed_data()

# from Plots import *
from PriceProcess import *
from WindProcess import *

from pyomo.environ import *



#------------------------------------------------------------------------------------------

def simulate_wind_and_price(data):
    # Simulate wind and price processes
    # Initialize a dictionary to store values of wind and price
    simulated_data = {'wind': {}, 'price': {}}

    # run wind model - wind(current, previous, data) starting from t=0 and we assume previous wind is 0
    for t in range(data['num_timeslots']):
        #Assuming wind at t-1 is 0
        if t == 0:
            simulated_data['wind'][t] = wind_model(data['wind_power_t_1'], data['wind_power_t_2'], data)
        else:
            if t == 1:
                simulated_data['wind'][t] = wind_model(simulated_data['wind'][t-1], data['wind_power_t_1'], data)
            else:
                simulated_data['wind'][t] = wind_model(simulated_data['wind'][t-1], simulated_data['wind'][t-2], data)

    # run price model - price(current, previous, wind, data) starting from t=0 and we assume previous price is 0
    for t in range(data['num_timeslots']):
        if t == 0:
            simulated_data['price'][t] = price_model(data['price_t_1'], data['price_t_2'], simulated_data['wind'][t], data)
        else:
            if t == 1:
                simulated_data['price'][t] = price_model(simulated_data['price'][t-1], data['price_t_1'], simulated_data['wind'][t], data)
            else:
                simulated_data['price'][t] = price_model(simulated_data['price'][t-1], simulated_data['price'][t-2], simulated_data['wind'][t], data)
    
    return simulated_data


def start_model(data, simulated_data):
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


def policy_model(data, run_times, simulated_data):

    policy1_results = []
    policy2_results = []
    for i in range(run_times):
        #print the run number
        print(f"{i}/{run_times}")

        #get generated wind and price
        simulated_data = simulate_wind_and_price(data)


        
        
        #------------------------------------------------------------------------------------------
        #define policies
        # policy 1: always turn off the electrolyzer
        model.policy1 = ConstraintList()
        for t in range(data['num_timeslots']):
            model.policy1.add(model.e[t] == 0)
            
        #solve the model for this policy
        # SOLVE
        # Create a solver
        solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured
        # Solve the model
        results = solver.solve(model, tee=True)

        # append the results to the list
        policy1_results.append(value(model.profit))

        # policy 2: if storage is less then 50% then H2p is 0
        model.policy2 = ConstraintList()
        for t in range(data['num_timeslots']):
            if value(model.S[t]) < 0.5*data['hydrogen_capacity']:
                model.policy2.add(model.H2P[t] == 0)

        #solve the model for this policy
        # SOLVE
        # Create a solver
        solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured
        # Solve the model
        results = solver.solve(model, tee=True)

        # append the results to the list
        policy1_results.append(value(model.profit))

        policy2_results.append(value(model.profit))

    # return each policy results average
    return sum(policy1_results)/run_times, sum(policy2_results)/run_times

policy1_results, policy2_results = policy_model(data, run_times)
print(f"Policy 1: {policy1_results}")
print(f"Policy 2: {policy2_results}")