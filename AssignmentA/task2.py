import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pyomo.environ as pyo
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


import sys
import networkx as nx
from pyomo.environ import *


# Import necessary modules
from Assignment_Codes.data import get_fixed_data
from Assignment_Codes.Plots import plot_results
from Assignment_Codes.PriceProcess import *
from Assignment_Codes.WindProcess import *
from task1 import *

#set a seed for reproducibility
#np.random.seed(42)

def generate_samples(data, node_values, num_samples):
    previous_wind, previous_price, current_wind, current_price  = node_values
    samples = []
    for _ in range(num_samples):
        w_next = wind_model(current_wind, previous_wind, data)
        p_next = price_model(current_price, previous_price, w_next, data)
        
        samples.append([w_next, p_next])
    
    return np.array(samples)

def cluster_samples(samples, num_clusters):
    # Standardize features
    scaler = StandardScaler()
    sample_matrix = scaler.fit_transform(samples)

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sample_matrix)
    
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert back to original values
    cluster_probs = np.bincount(labels) / len(labels)
    


    return cluster_centers, cluster_probs, labels

def visualize_clusters(samples, labels, cluster_centers):
    plt.figure(figsize=(10, 5))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_samples = samples[labels == label]  # Select samples in this cluster
        wind_vals, price_vals = cluster_samples[:, 0], cluster_samples[:, 1]
        plt.scatter(wind_vals, price_vals, label=f'Cluster {label}', alpha=0.6)

    # Plot centroids
    centroid_wind_vals, centroid_price_vals = cluster_centers[:, 0], cluster_centers[:, 1]
    plt.scatter(centroid_wind_vals, centroid_price_vals, marker='x', color='red', s=100, label='Centroids')

    plt.xlabel('Wind Power (MW)')
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Scenario Clustering')
    plt.legend()  # Add legend to identify clusters and centroids
    plt.show()

def scenario_tree(data, initial_node, look_ahead, num_samples, num_clusters):
    scenario_tree = {0: [(initial_node, 1.0)]}  # Start with the initial node with probability 1
    scenario_paths = {0: [[initial_node[2:]]]}  # Store paths for each scenario

    for t in range(1, look_ahead + 1):
        scenario_tree[t] = []
        scenario_paths[t] = []  # Initialize list for time step t

        for path_idx, (parent_node, parent_prob) in enumerate(scenario_tree[t - 1]):
            samples = generate_samples(data, parent_node, num_samples)
            clustered_samples, probabilities, _ = cluster_samples(samples, num_clusters)

            for i in range(num_clusters):
                new_values = clustered_samples[i].tolist()
                next_node = parent_node[2:] + new_values  # Shift previous values & add new ones
                new_prob = parent_prob * probabilities[i]

                scenario_tree[t].append((next_node, new_prob))

                # Extend the existing path with the new values
                new_path = scenario_paths[t - 1][path_idx] + [new_values]
                scenario_paths[t].append(new_path)
    # i am only interested in the last look_ahead values of the scenario path
    scenario_paths_final = scenario_paths[look_ahead]
    return scenario_tree, scenario_paths_final

def visualize_scenario_tree(scenario_tree, num_clusters):
    G = nx.DiGraph()
    pos = {}
    labels = {}
    
    for t, nodes in scenario_tree.items():
        for i, (node, prob) in enumerate(nodes):
            node_id = f"{t}_{i}"
            G.add_node(node_id, pos=(t, i))
            wind_price_values = [int(value) for value in node]  # Format wind and price values as integers
            labels[node_id] = f"{wind_price_values}\n{int(prob * 100)}%"  # Format probability as integer percentage
            if t > 0:
                parent_node_id = f"{t-1}_{i // num_clusters}"
                G.add_edge(parent_node_id, node_id)
    
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=100, node_color='skyblue', font_size=7, font_weight='bold', arrows=True)
    plt.title('Scenario Tree')
    plt.show()

def print_matrix(variable, variable_name, model):
    print(f"{variable_name}")
    matrix = []
    for s in model.S:
        row = []
        for t in model.T:
            row.append(variable[s, t].value)
        matrix.append(row)

    # Print the matrix
    for row in matrix:
        print(" ".join(f"{val:.2f}" if val is not None else "None" for val in row))

def calculate_max_look_ahead(V, B):
    L = 1  # Start with L = 1
    while V * L * B**(L - 1) <= 1000:
        L += 1
    return L - 1 
 
def stochastic_optimization_here_and_now(data, wind, price, hydrogen_stock, ele, tau, T, n_clusters=2, num_samples=20):
    if tau == 0:
        initial_node = [data['wind_power_t_1'], data['price_t_1'], wind[tau], price[tau]]
    else:
        initial_node = [wind[tau-1], price[tau-1], wind[tau], price[tau]]
    
    # Get the number of tlook ahead days
    look_ahead = calculate_max_look_ahead(3, n_clusters)  # Assuming 3 binary variables (y_on, y_off, ele) and 2 clusters for this example

    # Determine number of days to optimize
    num_of_days = max(0, min(look_ahead, T - tau -1))  # Ensure non-negative
    
    # Load predicted scenario tree
    scenario_tree_matrix, scenario_paths_matrix = scenario_tree(data, initial_node, num_of_days, num_samples, n_clusters)
    
    # the scenario probabilities will be the last look_ahead value
    probabilities = [prob for node, prob in scenario_tree_matrix[num_of_days]]

    # Number of scenarios
    n_scenarios = len(probabilities)
        
    # Initialize Gurobi model
    # Create a model
    model = ConcreteModel()


    # Define sets
    model.T = RangeSet(0, num_of_days)
    model.S = RangeSet(0, n_scenarios - 1)

    # Declare variables for each scenario and time period
    model.e = Var(model.S, model.T, within=Binary)
    model.P2H = Var(model.S, model.T, bounds=(0, data['p2h_rate']))
    model.Storage = Var(model.S, model.T, bounds=(0, data['hydrogen_capacity']))
    model.H2P = Var(model.S, model.T, bounds=(0, data['h2p_rate']))
    model.p = Var(model.S, model.T, bounds=(0, None))
    model.yon = Var(model.S, model.T, within=Binary)
    model.yoff = Var(model.S, model.T, within=Binary)
    #------------------------------------------------------------------------------------------

    
    # Define objective function (expected cost minimization)
    model.cost = Objective(
        expr=sum(probabilities[s]*
                 sum(model.e[s,t] * data['electrolyzer_cost'] + model.p[s,t] * scenario_paths_matrix[s][t][1] for t in model.T) for s in model.S), #scenario_paths_matrix[scenario][time][(wind, price)]
        sense=minimize
    )

    # Constraint on available power always == demand
    model.Power = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Power.add(model.p[s,t] + scenario_paths_matrix[s][t][0] + model.H2P[s,t]*data['conversion_h2p'] - model.P2H[s,t] >= data['demand_schedule'][tau+t])
            #print('scnario:', s, 'time:', t, 'wind:', scenario_paths_matrix[s][t][0], 'price:', scenario_paths_matrix[s][t][1], 'demand:', data['demand_schedule'][t])                
    
    # there is a conversion rate from power to hydrogen
    model.P2H_Conversion = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.P2H_Conversion.add(model.P2H[s,t] <= model.e[s,t] * data['p2h_rate'])  

    # thre is a conversion rate from hydrogen to power
    model.H2P_Conversion = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.H2P_Conversion.add(model.H2P[s,t] <= model.Storage[s,t])

    #  When an amount of hydrogen is produced at t-1, it becomes stored and available 
    # in the tank from the next timeslot t, plus the amount of hydrogen already stored (t-1) 
    # and minus the amount of hydrogen turned back into power in the same period t
    model.Storage_Constraint = ConstraintList()
    for s in model.S:
        for t in model.T:
            if t == 0:
                model.Storage_Constraint.add(model.Storage[s,t] == hydrogen_stock[tau])
            else:
                model.Storage_Constraint.add(model.Storage[s,t] == model.Storage[s,t-1] + model.P2H[s,t-1] * data['conversion_p2h']  - model.H2P[s,t-1])

    # if the electrolyzer is set to on we cannot turn it off and viceversa
    model.Electrolyzer = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(model.yon[s,t] + model.yoff[s,t] <= 1)

    # we can only turn it off if it is on
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(model.e[s,t] >= model.yoff[s,t])

    # we can only turn it on if it is off
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(model.e[s,t] <= 1- model.yon[s,t])

    # there is also relatioship between ele at time t and ele at time t-1 and yon and yoff at time t-1
    for s in model.S:
        for t in model.T:
            if t == 0:
                model.Electrolyzer.add(model.e[s,t] == ele[tau])
            else:
                model.Electrolyzer.add(model.e[s,t] == model.e[s,t-1] + model.yon[s,t-1] - model.yoff[s,t-1])

    # Non-anticipativity constraints. If two scenarios share history at day t, then the decisions made at day t for both scenarios must be the same
    model.NonAnticipativity = ConstraintList()
    for s in model.S:
        for t in model.T:
            for s2 in model.S:
                if scenario_paths_matrix[s][t] == scenario_paths_matrix[s2][t]: #s != s2 and 
                    model.NonAnticipativity.add(model.e[s, t] == model.e[s2, t])
                    model.NonAnticipativity.add(model.P2H[s, t] == model.P2H[s2, t])
                    model.NonAnticipativity.add(model.Storage[s, t] == model.Storage[s2, t])
                    model.NonAnticipativity.add(model.H2P[s, t] == model.H2P[s2, t])
                    model.NonAnticipativity.add(model.p[s, t] == model.p[s2, t])
                    model.NonAnticipativity.add(model.yon[s, t] == model.yon[s2, t])
                    model.NonAnticipativity.add(model.yoff[s, t] == model.yoff[s2, t])


    # Solve the model
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=False)
    # return the decisions for the first time period for each scenario
    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        print(f"Optimal solution found, objective value: {model.cost()}") 

    
    # Print the matrix
    #print_matrix(model.e, "Electrolyzer", model)
    #print_matrix(model.P2H, "P2H", model)
    #print_matrix(model.Storage, "Storage", model)
    #print_matrix(model.H2P, "H2P", model)
    #print_matrix(model.p, "Power", model)
    #print_matrix(model.yon, "Yon", model)
    #print_matrix(model.yoff, "Yoff",model)

    #return model.e[0,0].value, model.P2H[0,0].value, model.Storage[0,0].value, model.H2P[0,0].value, model.p[0,0].value, model.yon[0,0].value, model.yoff[0,0].value
    # we return the decisions for the first time period for each scenario
    # we can do [0,0] because the decision in this period is the same for all scenarios
    return model.yon[0,0].value, model.yoff[0,0].value, model.P2H[0,0].value, model.H2P[0,0].value, model.p[0,0].value

def generate_expected_scenario_path(data, wind, price, tau, T, num_samples=20):
    """
    Generates a deterministic scenario path from tau using expected values from the stochastic models.

    Returns:
        List of (wind, price) pairs starting from tau for use in the EV policy.
    """
    horizon = T - tau
    scenario_path = []

    # Step 1: Initialize with current values
    wind_curr = wind[tau]
    price_curr = price[tau]
    scenario_path.append((wind_curr, price_curr))

    # Step 2: Use appropriate past values
    if tau == 0:
        wind_prev = data['wind_power_t_1']
        price_prev = data['price_t_1']
    else:
        wind_prev = wind[tau - 1]
        price_prev = price[tau - 1]

    # Step 3: Generate expected path
    for t in range(1, horizon):
        # Build node input for generator: [w_prev, p_prev, w_curr, p_curr]
        node = [wind_prev, price_prev, wind_curr, price_curr]
        samples = generate_samples(data, node, num_samples)
        w_next = np.mean(samples[:, 0])
        p_next = np.mean(samples[:, 1])

        scenario_path.append((w_next, p_next))

        # Shift forward for next iteration
        wind_prev, price_prev=  wind_curr, price_curr
        wind_curr, price_curr = w_next, p_next

    return scenario_path

def EV_stochastic_optimization_here_and_now(data, wind, price, hydrogen_stock, ele, tau, T, n_clusters=2, num_samples=20):

    # Generate expected scenario path
    scenario_path = generate_expected_scenario_path(data, wind, price, tau, T, num_samples)
    
    # Get the number of tlook ahead days
    look_ahead = calculate_max_look_ahead(3, n_clusters)  # Assuming 3 binary variables (y_on, y_off, ele) and 2 clusters for this example

    # Determine number of days to optimize
    num_of_days = max(0, min(look_ahead, T - tau -1))  # Ensure non-negative
        

    # Number of scenarios
    n_scenarios = 1
        
    # Initialize Gurobi model
    # Create a model
    model = ConcreteModel()


    # Define sets
    model.T = RangeSet(0, num_of_days)
    model.S = RangeSet(0, n_scenarios - 1)

    # Declare variables for each scenario and time period
    model.e = Var(model.S, model.T, within=Binary)
    model.P2H = Var(model.S, model.T, bounds=(0, data['p2h_rate']))
    model.Storage = Var(model.S, model.T, bounds=(0, data['hydrogen_capacity']))
    model.H2P = Var(model.S, model.T, bounds=(0, data['h2p_rate']))
    model.p = Var(model.S, model.T, bounds=(0, None))
    model.yon = Var(model.S, model.T, within=Binary)
    model.yoff = Var(model.S, model.T, within=Binary)
    #------------------------------------------------------------------------------------------

    
    # Define objective function (expected cost minimization)
    model.cost = Objective(
        expr=sum(sum(model.e[s,t] * data['electrolyzer_cost'] + model.p[s,t] * scenario_path[t][1] for t in model.T) for s in model.S), #scenario_paths_matrix[scenario][time][(wind, price)]
        sense=minimize
    )

    # Constraint on available power always == demand
    model.Power = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Power.add(model.p[s,t] + scenario_path[t][0] + model.H2P[s,t]*data['conversion_h2p'] - model.P2H[s,t] >= data['demand_schedule'][tau+t])
            #print('scnario:', s, 'time:', t, 'wind:', scenario_paths_matrix[s][t][0], 'price:', scenario_paths_matrix[s][t][1], 'demand:', data['demand_schedule'][t])                
    
    # there is a conversion rate from power to hydrogen
    model.P2H_Conversion = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.P2H_Conversion.add(model.P2H[s,t] <= model.e[s,t] * data['p2h_rate'])  

    # thre is a conversion rate from hydrogen to power
    model.H2P_Conversion = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.H2P_Conversion.add(model.H2P[s,t] <= model.Storage[s,t])

    #  When an amount of hydrogen is produced at t-1, it becomes stored and available 
    # in the tank from the next timeslot t, plus the amount of hydrogen already stored (t-1) 
    # and minus the amount of hydrogen turned back into power in the same period t
    model.Storage_Constraint = ConstraintList()
    for s in model.S:
        for t in model.T:
            if t == 0:
                model.Storage_Constraint.add(model.Storage[s,t] == hydrogen_stock[tau])
            else:
                model.Storage_Constraint.add(model.Storage[s,t] == model.Storage[s,t-1] + model.P2H[s,t-1] * data['conversion_p2h']  - model.H2P[s,t-1])

    # if the electrolyzer is set to on we cannot turn it off and viceversa
    model.Electrolyzer = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(model.yon[s,t] + model.yoff[s,t] <= 1)

    # we can only turn it off if it is on
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(model.e[s,t] >= model.yoff[s,t])

    # we can only turn it on if it is off
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(model.e[s,t] <= 1- model.yon[s,t])

    # there is also relatioship between ele at time t and ele at time t-1 and yon and yoff at time t-1
    for s in model.S:
        for t in model.T:
            if t == 0:
                model.Electrolyzer.add(model.e[s,t] == ele[tau])
            else:
                model.Electrolyzer.add(model.e[s,t] == model.e[s,t-1] + model.yon[s,t-1] - model.yoff[s,t-1])


    # Solve the model
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=False)
    # return the decisions for the first time period for each scenario
    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        print(f"Optimal solution found, objective value: {model.cost()}") 

    
    # Print the matrix
    #print_matrix(model.e, "Electrolyzer", model)
    #print_matrix(model.P2H, "P2H", model)
    #print_matrix(model.Storage, "Storage", model)
    #print_matrix(model.H2P, "H2P", model)
    #print_matrix(model.p, "Power", model)
    #print_matrix(model.yon, "Yon", model)
    #print_matrix(model.yoff, "Yoff",model)

    #return model.e[0,0].value, model.P2H[0,0].value, model.Storage[0,0].value, model.H2P[0,0].value, model.p[0,0].value, model.yon[0,0].value, model.yoff[0,0].value
    # we return the decisions for the first time period for each scenario
    # we can do [0,0] because the decision in this period is the same for all scenarios
    return model.yon[0,0].value, model.yoff[0,0].value, model.P2H[0,0].value, model.H2P[0,0].value, model.p[0,0].value


# try with multistage stochastic optimization
if __name__ == "__main__":
    data = get_fixed_data()
    #generate experiments
    experiments = generate_experiments(data['num_timeslots'], n=1)

    #evaluate policy
    results = evaluate_policy_over_experiments(stochastic_optimization_here_and_now, data, experiments, data['num_timeslots'], verbose=True)
    print("stochastic_optimization_here_and_now policy costs: ", results)

    results = evaluate_policy_over_experiments(EV_stochastic_optimization_here_and_now, data, experiments, data['num_timeslots'], verbose=True)
    print("EV policy costs: ", results)