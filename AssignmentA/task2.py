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

sys.path.append(os.path.abspath('C:/Users/luisa/OneDrive - Danmarks Tekniske Universitet/DTU/2year_2semester/Decision-making under uncertainty/AssignmentA/Assignment_Codes'))

# Import necessary modules
from Assignment_Codes.data import get_fixed_data
from Assignment_Codes.Plots import plot_results
from Assignment_Codes.PriceProcess import *
from Assignment_Codes.WindProcess import *
from task1_feasibility_check import check_feasibility
from task1_policy import make_decision
from task1_nextstate import nextstate

#set a seed for reproducibility
np.random.seed(42)

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

def stochastic_optimization_here_and_now(data,T, tau, initial_node, current_storage, current_ele_status, look_ahead, num_samples, num_clusters):
    # Determine number of days to optimize
    num_of_days = max(0, min(look_ahead, T - tau))
    
    # Load predicted scenario tree
    scenario_tree_matrix, scenario_paths_matrix = scenario_tree(data, initial_node, num_of_days, num_samples, num_clusters)
    
    # the scenario probabilities will be the last look_ahead value
    probabilities = [prob for node, prob in scenario_tree_matrix[num_of_days]]

    # Number of scenarios
    n_scenarios = len(probabilities)
        
    # Initialize Gurobi model
    # Create a model
    model = ConcreteModel()


    # Define sets
    model.T = RangeSet(0, num_of_days - 1)
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
    model.profit = Objective(
        expr=sum(probabilities[s]*
                 sum(model.e[s,t] * data['electrolyzer_cost'] + model.p[s,t] * scenario_paths_matrix[s][t][1] for t in model.T) for s in model.S), #scenario_paths_matrix[scenario][time][(wind, price)]
        sense=minimize
    )

    # Constraint on available power always == demand
    model.Power = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Power.add(model.p[s,t] + scenario_paths_matrix[s][t][0] + model.H2P[s,t]*data['conversion_h2p'] - model.P2H[s,t] >= data['demand_schedule'][t])
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
                model.Storage_Constraint.add(model.Storage[s,t] == current_storage)
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
                model.Electrolyzer.add(model.e[s,t] == current_ele_status)
            else:
                model.Electrolyzer.add(model.e[s,t] == model.e[s,t-1] + model.yon[s,t-1] - model.yoff[s,t-1])

    # Non-anticipativity constraints. If two scenarios share history at day t, then the decisions made at day t for both scenarios must be the same
    model.NonAnticipativity = ConstraintList()
    for t in model.T:
    #    if t == 0:
    #        continue
        for s in model.S:
            for s2 in model.S:
                if s != s2 and scenario_paths_matrix[s][t] == scenario_paths_matrix[s2][t]:
                    model.NonAnticipativity.add(model.e[s, t] == model.e[s2, t])
                    model.NonAnticipativity.add(model.P2H[s, t] == model.P2H[s2, t])
                    model.NonAnticipativity.add(model.Storage[s, t] == model.Storage[s2, t])
                    model.NonAnticipativity.add(model.H2P[s, t] == model.H2P[s2, t])
                    model.NonAnticipativity.add(model.p[s, t] == model.p[s2, t])
                    model.NonAnticipativity.add(model.yon[s, t] == model.yon[s2, t])
                    model.NonAnticipativity.add(model.yoff[s, t] == model.yoff[s2, t])


    # Solve the model
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=True)
    # return the decisions for the first time period for each scenario
    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found")

    
    # Print the matrix
    #print_matrix(model.e, "Electrolyzer", model)
    #print_matrix(model.P2H, "P2H", model)
    #print_matrix(model.Storage, "Storage", model)
    #print_matrix(model.H2P, "H2P", model)
    print_matrix(model.p, "Power", model)
    #print_matrix(model.yon, "Yon", model)
    #print_matrix(model.yoff, "Yoff",model)

    #return model.e[0,0].value, model.P2H[0,0].value, model.Storage[0,0].value, model.H2P[0,0].value, model.p[0,0].value, model.yon[0,0].value, model.yoff[0,0].value
    # we return the decisions for the first time period for each scenario
    # we can do [0,0] because the decision in this period is the same for all scenarios
    return model.P2H[0,0].value, model.H2P[0,0].value, model.p[0,0].value, model.yon[0,0].value, model.yoff[0,0].value


def calculate_max_look_ahead(V, B):
    def equation(L, V, B):
        return V * L * B**(L-1) - 1000

    L_initial_guess = 1
    L_solution = fsolve(equation, L_initial_guess, args=(V, B))
    return int(L_solution[0])
 
def multi_stage_sp_policy(data, T, num_clusters, num_samples=500):

    # calculate the number of look ahead days possible
    # given V∗𝐿∗𝐵*^(𝐿−1) < 1000
    # where V is the number of variables, L is the number of look ahead days, B is the number of scenarios, and L is the number of time periods
    # calculate the number of look ahead days possible
    # just binary variables
    V = 3
    B = num_clusters
    #look_ahead = calculate_max_look_ahead(V, B)
    look_ahead = 3  
    print(f"Maximum number of look-ahead days: {look_ahead}")


    # Initialize variables
    e = np.zeros(T)
    P2H = np.zeros(T)
    S = np.zeros(T)
    H2P = np.zeros(T)
    p = np.zeros(T)
    yon = np.zeros(T)
    yoff = np.zeros(T)

    wind_trajectory = []
    price_trajectory = []

    for tau in range(T):
        if tau == 0:
            # Initialize state variables
            previous_wind = data['wind_power_t_2']
            previous_price = data['price_t_2']
            current_wind = data['wind_power_t_1']
            current_price = data['price_t_1']
            current_storage = 10
            current_ele_status = 1



        # reveal uncertainity the wind and price values
        wind_next = wind_model(current_wind, previous_wind, data)
        price_next = price_model(current_price, previous_price, wind_next, data)

        # store the revealed values
        wind_trajectory.append(wind_next)
        price_trajectory.append(price_next)

        # store ele and storage values
        e[tau] = current_ele_status
        S[tau] = current_storage
        
        parent_node = [current_wind, current_price, wind_next, price_next]
        # generate scenarios based on the revealed values
        # reduce scnerios
        # solve the optimization problem
        # implement the decision
        #e[tau], P2H[tau], S[tau], H2P[tau], p[tau], yon[tau], yoff[tau] = stochastic_optimization_here_and_now(data, T, tau, parent_node, current_storage, current_ele_status, look_ahead, num_samples, num_clusters)
        P2H[tau], H2P[tau], p[tau], yon[tau], yoff[tau] = stochastic_optimization_here_and_now(data, T, tau, parent_node, current_storage, current_ele_status, look_ahead, num_samples, num_clusters)
        

        # check if the policy is feasible
        #status, yon[tau], yoff[tau], P2H[tau], H2P[tau], p[tau] = check_feasibility(
        #    data, yon[tau], yoff[tau], P2H[tau], H2P[tau], p[tau], wind_next, data['demand_schedule'][tau], current_storage, current_ele_status
        #    )
#
        #if not status:
        #    print(f"Policy not feasible at time {tau}")
        #    #lets define a feasible policy
        #    #status, yon[tau], yoff[tau], P2H[tau], H2P[tau], p[tau] = correct_decision(data, yon[tau], yoff[tau], P2H[tau], H2P[tau], p[tau], wind_next, data['demand_schedule'][tau], current_storage, current_ele_status)
        #    break
        
        # update state variables for the next iteration
        current_storage = current_storage + P2H[tau] * data['conversion_p2h'] - H2P[tau]
        current_ele_status = current_ele_status + yon[tau] - yoff[tau]

        # Update current values for next iteration
        previous_wind = current_wind
        previous_price = current_price
        current_wind = wind_next
        current_price = price_next

    # 5. Create a 'times' array for your x-axis
    times = np.arange(tau+1)

     # 4. Store the solution arrays in a dictionary that matches plot_results requirements
    results_dict = {
        'electrolyzer_status':       e[:tau + 1],
        'hydrogen_storage_level':    S[:tau + 1],
        'power_to_hydrogen':        P2H[:tau + 1],
        'hydrogen_to_power':        H2P[:tau + 1],
        'grid_power':               p[:tau + 1]
    }


    # 6. Finally, call plot_results
    plot_results(
        data              = data,
        times             = times,
        wind_trajectory   = wind_trajectory,
        demand_schedule   = data['demand_schedule'][:tau + 1],
        results           = results_dict,
        price_trajectory  = price_trajectory
    )

    return e, P2H, S, H2P, p, yon, yoff





#look_ahead = 3
#num_samples = 20
num_clusters = 2
#node_values = [5,30, 4, 28]
T = 10
#tau = 5
#current_storage = 10
#current_ele_status = 1


data = get_fixed_data()
#samples = generate_samples(data, node_values, num_samples)
#cluster_centers, cluster_probs, labels = cluster_samples(samples, num_clusters)
#visualize_clusters(samples, labels, cluster_centers)
#model = stochastic_optimization(cluster_centers, cluster_probs)
#scenario_tree = scenario_tree(data, node_values, look_ahead, num_samples, num_clusters)
#visualize_scenario_tree(scenario_tree, num_clusters)

#decision_now = stochastic_optimization_here_and_now(data, T, tau, node_values, current_storage, current_ele_status, look_ahead, num_samples, num_clusters)

e, P2H, S, H2P, p, yon, yoff = multi_stage_sp_policy(data, T, num_clusters)