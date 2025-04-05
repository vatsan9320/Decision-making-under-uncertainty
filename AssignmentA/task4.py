from Assignment_Codes.data import get_fixed_data
from task1_main import evaluate_policy_over_experiments
from task0 import simulate_wind_and_price
from task1_policy import make_dummy_decision
from task0 import solve_optimal_in_hindsight
from task2 import *
from task3_luisa_main import *
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np


#add seed for reproducibility
#np.random.seed(42)

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



# stochastic programming policy
num_clusters = [ 3, 4, 5]
for num_cluster in num_clusters:
    results[f"SE_{num_cluster}"] = evaluate_policy_over_experiments(stochastic_optimization_here_and_now, data, experiments, T=data['num_timeslots'], n_clusters=num_cluster)

# stochastic programming policy with the expectedd value
results["SE_EV"] = evaluate_policy_over_experiments(EV_stochastic_optimization_here_and_now, data, experiments, T=data['num_timeslots'], n_clusters=1)


# ADP
# Train value function approximation (outside the experiment loop!)
theta = train_value_function(data, gamma=0.95, I=50, K=5)
# Evaluate ADP policy using the same experiments
results["ADP"] = evaluate_policy_over_experiments(adp_policy_wrapper(theta), data, experiments, T=data['num_timeslots'])



def plot_cost_histograms(results):
    """
    Plot histogram of costs per policy across experiments.
    """
    num_policies = len(results)
    fig, axes = plt.subplots(1, num_policies, figsize=(5 * num_policies, 5), sharey=True)

    if num_policies == 1:
        axes = [axes]  # make iterable

    for ax, (name, costs) in zip(axes, results.items()):
        valid_costs = [c for c in costs if not np.isnan(c) and not np.isinf(c)]
        avg_cost = np.mean(valid_costs) if valid_costs else float('nan')

        ax.hist(valid_costs, bins=8, color='cornflowerblue', edgecolor='black')
        ax.set_title(f"{name}\nAvg Cost: {avg_cost:.1f}")
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Cost Distribution Across 20 Experiments", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_cost_histograms(results)


