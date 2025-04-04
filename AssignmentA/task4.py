from Assignment_Codes.data import get_fixed_data
from task1_main import evaluate_policy_over_experiments
from task0 import simulate_wind_and_price
from task1_policy import make_dummy_decision
from task0 import solve_optimal_in_hindsight
from task2 import stochastic_optimization_here_and_now
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np


#add seed for reproducibility
np.random.seed(42)

# Load given data
data = get_fixed_data()

def generate_experiments(num_timeslots, n):
    experiments = []
    for _ in range(n):
        sim = simulate_wind_and_price(data, T=num_timeslots)
        wind_seq = [sim['wind'][t] for t in range(num_timeslots)]
        price_seq = [sim['price'][t] for t in range(num_timeslots)]
        experiments.append((wind_seq, price_seq))
    return experiments




experiments = generate_experiments(num_timeslots=data['num_timeslots'], n=2)
results = {}

# OIH — inline loop
# we need to pass the data and the wind and price sequences to the function
oih_costs = []
for wind_seq, price_seq in experiments:
    cost, _ = solve_optimal_in_hindsight(data, wind_seq, price_seq, T=data['num_timeslots'])
    oih_costs.append(cost)

results["OIH"] = oih_costs


# Dummy policy
results["Dummy"] = evaluate_policy_over_experiments(make_dummy_decision, data, experiments, T=data['num_timeslots'])




# Multi-stage
results["Multi-stage"] = evaluate_policy_over_experiments(stochastic_optimization_here_and_now, data, experiments, T=data['num_timeslots'])

# ADP
#theta = train_value_function(data, gamma=0.95, I=50, K=5)
#results["ADP"] = evaluate_policy_over_experiments(adp_policy_wrapper(theta), data, experiments)



def plot_per_experiment_comparison(results):
    policy_names = list(results.keys())
    num_experiments = len(next(iter(results.values())))
    
    # Matrix shape: (num_policies, num_experiments)
    cost_matrix = []
    for name in policy_names:
        costs = results[name]
        if not isinstance(costs, (list, np.ndarray)):
            costs = [costs] * num_experiments
        row = []
        for c in costs:
            try:
                row.append(float(c))
            except:
                row.append(np.nan)
        cost_matrix.append(row)

    cost_matrix = np.array(cost_matrix)
    x = np.arange(num_experiments)
    bar_width = 0.15

    plt.figure(figsize=(14, 6))
    colors = plt.get_cmap("tab10").colors  # better palette

    for i, name in enumerate(policy_names):
        plt.bar(x + i * bar_width, cost_matrix[i], width=bar_width, label=name, color=colors[i % len(colors)])


    plt.xlabel("Experiment")
    plt.ylabel("Total Cost")
    plt.title("Policy Cost per Experiment")
    plt.xticks(x + bar_width * (len(policy_names) - 1) / 2, [str(i) for i in x])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_per_experiment_comparison(results)

