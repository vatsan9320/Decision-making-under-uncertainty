import sys
import os

# Add the directory to the Python path
sys.path.append(os.path.abspath('C:/Users/luisa/OneDrive - Danmarks Tekniske Universitet/DTU/2year_2semester/Decision-making under uncertainty/AssignmentA/Assignment_Codes'))
#sys.path.append(os.path.abspath('../Decision-making under uncertainty/AssignmentA/Assignment_Codes'))

# import specific modules from folder AssignmentA/Assignment_Codes
from Assignment_Codes.data import get_fixed_data
from Assignment_Codes.PriceProcess import *
from Assignment_Codes.WindProcess import *
from Assignment_Codes.Plots import plot_results

# given data
data = get_fixed_data()

# from Plots import *
from pyomo.environ import *


def simulate_wind_and_price(data, T):
    """

    Args:
        data (dict): Must include wind_power_t_1, wind_power_t_2, price_t_1, price_t_2
        T (int): Number of time steps to simulate (defaults to T)

    Returns:
        dict: {
            'wind': {t: wind at t for t in 0..T-1},
            'price': {t: price at t for t in 0..T-1}
        }
    """

    # Initialize history
    wind_history = [data['wind_power_t_2'], data['wind_power_t_1']]
    price_history = [data['price_t_2'], data['price_t_1']]

    wind = {}
    price = {}

    for t in range(T):
        w_next = wind_model(wind_history[-1], wind_history[-2], data)
        wind_history.append(w_next)
        wind[t] = w_next

        p_next = price_model(price_history[-1], price_history[-2], w_next, data)
        price_history.append(p_next)
        price[t] = p_next

    return {'wind': wind, 'price': price}




def solve_optimal_in_hindsight(data, wind_seq, price_seq, T):
    # Create a model
    model = ConcreteModel()

    #------------------------------------------------------------------------------------------
    # Declare variables
    #  The electrolyzer can be ON or OFF at each hour t
    model.e = Var(range(T), within=Binary)
    # the amount of power to turn into hydrogen
    model.P2H = Var(range(T), bounds=(0, data['p2h_rate']))
    # the amount of hydrogen stored
    model.S = Var(range(T), bounds=(0, data['hydrogen_capacity']))
    # the amount of hydrogen turned back into power
    model.H2P = Var(range(T), bounds=(0, data['h2p_rate']))
    # power to take from the grid - change to be the power to take from the grid straigth to the industrial plant
    model.p = Var(range(T), bounds=(0, None))
    # binary if electrolizer can be turn on or not
    model.yon = Var(range(T), within=Binary)
    # binary if electrolizer can be turn off or not
    model.yoff = Var(range(T), within=Binary)
    #------------------------------------------------------------------------------------------



    #------------------------------------------------------------------------------------------
    # Objective function: Minimization of costs
    model.cost = Objective(
        expr=sum(model.e[t] * data['electrolyzer_cost'] + model.p[t] * price_seq[t] for t in range(T)),   
        sense=minimize)
    #------------------------------------------------------------------------------------------



    # Constraints
    #------------------------------------------------------------------------------------------
    # set initial status of the electrolyzer
    model.InitialStatus = ConstraintList()
    model.InitialStatus.add(model.e[0] == 0)  
    model.InitialStatus.add(model.S[0] == 0)

    # Constraint on available power always == demand
    model.Power = ConstraintList()
    for t in range(T):
        # power in the fabric = power from the grid + power from the wind + power from the hydrogen = demand
        # TODO: should be == ??? why is it infeasible? because we cannot stop wind
        model.Power.add(model.p[t] + wind_seq[t] + model.H2P[t]*data['conversion_h2p'] - model.P2H[t] >= data['demand_schedule'][t])
                        

    # there is a conversion rate from power to hydrogen
    model.P2H_Conversion = ConstraintList()
    # we convert if the electrolyzer is on and if we had surplus power from wind in t-1, meaning wind > demand
    # doesnt need to be equal because maybe we want to convert less than the surplus
    #for t in range(T):
    for t in range(T):
        model.P2H_Conversion.add(model.P2H[t] <= model.e[t] * data['p2h_rate'])  
    #model.P2H_Conversion.add(model.P2H[t] <= model.e[t-1] * data['p2h_rate']) 

    # thre is a conversion rate from hydrogen to power
    model.H2P_Conversion = ConstraintList()
    for t in range(T):
        model.H2P_Conversion.add(model.H2P[t] <= model.S[t])

    #  When an amount of hydrogen is produced at t-1, it becomes stored and available 
    # in the tank from the next timeslot t, plus the amount of hydrogen already stored (t-1) 
    # and minus the amount of hydrogen turned back into power in the same period t
    model.Storage = ConstraintList()
    for t in range(1,T):
        model.Storage.add(model.S[t] == model.S[t-1] + model.P2H[t-1] * data['conversion_p2h']  - model.H2P[t-1])


    # if the electrolyzer is set to on we cannot turn it off and viceversa
    model.Electrolyzer = ConstraintList()
    for t in range(T):
        model.Electrolyzer.add(model.yon[t] + model.yoff[t] <= 1)

    # we can only turn it off if it is on
    for t in range(T):
        model.Electrolyzer.add(model.e[t] >= model.yoff[t])

    # we can only turn it on if it is off
    for t in range(T):
        model.Electrolyzer.add(model.e[t] <= 1- model.yon[t])

    # there is also relatioship between ele at time t and ele at time t-1 and yon and yoff at time t-1
    for t in range(1, T):
        model.Electrolyzer.add(model.e[t] == model.e[t-1] + model.yon[t-1] - model.yoff[t-1])

    #------------------------------------------------------------------------------------------


    #------------------------------------------------------------------------------------------
    # SOLVE
    # Create a solver
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

    # Solve the model
    results = solver.solve(model, tee=False)

    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        print(f"Optimal solution found, objective value: {model.cost()}") 

        # Print out variable values and objective value
        #print("Variable values:")
        #for t in range(T):
        #    print(f"u[{t}]: {value(model.e[t]):.3f}")
        #    print(f"P2H[{t}]: {value(model.P2H[t]):.3f}")
        #    print(f"H[{t}]: {value(model.S[t]):.3f}")
        #    print(f"H2P[{t}]: {value(model.H2P[t]):.3f}")
        #    print(f"p[{t}]: {value(model.p[t]):.3f}")
        #print(f"\nObjective value: {value(model.cost):.3f}\n")
    else:
        print("No optimal solution found")
    #------------------------------------------------------------------------------------------




    #------------------------------------------------------------------------------------------
    # plot the results
    # create a dictionary to store the results

    results = {'electrolyzer_status': [value(model.e[t]) for t in range(T)],
                'power_to_hydrogen': [value(model.P2H[t]) for t in range(T)],
                'hydrogen_storage_level': [value(model.S[t]) for t in range(T)],
                'hydrogen_to_power': [value(model.H2P[t]) for t in range(T)],
                'grid_power': [value(model.p[t]) for t in range(T)]}   

    price_trajectory = [price_seq[t] for t in range(T)]
    wind_trajectory = [wind_seq[t] for t in range(T)]

    # run Plots.py
    # Plot the results using the function from Plots.py


    #return the objective value
    return value(model.cost), results

    #------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sim_data = simulate_wind_and_price(data, T=data['num_timeslots'])
    wind_seq = sim_data['wind']
    price_seq = sim_data['price']

    # Convert dictionaries to lists
    wind_trajectory = [wind_seq[t] for t in range(data['num_timeslots'])]
    price_trajectory = [price_seq[t] for t in range(data['num_timeslots'])]

    obj, results = solve_optimal_in_hindsight(data, wind_seq, price_seq, T=data['num_timeslots'])

    plot_results(
        data=data,
        times=range(data['num_timeslots']),
        wind_trajectory=wind_trajectory,  # Pass the converted list
        demand_schedule=data['demand_schedule'],
        results=results,
        price_trajectory=price_trajectory  # Pass the converted list
    )