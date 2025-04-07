from Assignment_Codes.data import get_fixed_data
from task1 import evaluate_policy_over_experiments
from task1_dummy import make_dummy_decision
from task0 import solve_optimal_in_hindsight
from task2 import *
from task3 import *
import numpy as np
import matplotlib.pyplot as plt
import json

# Set seed for reproducibility
np.random.seed(42)

# Load given data
data = get_fixed_data()

# -------- JSON Fix Helper ----------
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj

# -------- Generate Experiments --------
#def generate_experiments(num_timeslots, n):
    experiments = []
    for _ in range(n):
        sim = simulate_wind_and_price(data, T=num_timeslots)
        wind_seq = [sim['wind'][t] for t in range(num_timeslots)]
        price_seq = [sim['price'][t] for t in range(num_timeslots)]
        experiments.append((wind_seq, price_seq))
    return experiments
#
#experiments = generate_experiments(num_timeslots=data['num_timeslots'], n=20)
#
#with open('experiments.json', 'w') as f:
#    json.dump(make_json_serializable(experiments), f)
#
results = {}

#load experiments from file
with open('experiments.json', 'r') as f:
    experiments = json.load(f)

# -------- OIH Policy --------
#print("------------------------------------------------> Running experiments: OIH")
#oih_costs = []
#for wind_seq, price_seq in experiments:
#    cost, _ = solve_optimal_in_hindsight(data, wind_seq, price_seq, T=data['num_timeslots'])
#    oih_costs.append(cost)
#results["OIH"] = oih_costs
#with open('results_OIH.json', 'w') as f:
#    json.dump(make_json_serializable({"OIH": oih_costs}), f)
#
## -------- Dummy Policy --------
#print("------------------------------------------------> Running experiments: Dummy policy")
#dummy_costs = evaluate_policy_over_experiments(make_dummy_decision, data, experiments, T=data['num_timeslots'])
#results["Dummy"] = dummy_costs
#with open('results_Dummy.json', 'w') as f:
#    json.dump(make_json_serializable({"Dummy": dummy_costs}), f)
#
## -------- Stochastic Programming Policies --------
#num_clusters = [3, 4, 5]
#for num_cluster in num_clusters:
#    key = f"SE_{num_cluster}"
#    print(f"------------------------------------------------> Running experiments: {key}")
#    se_costs = evaluate_policy_over_experiments(
#        stochastic_optimization_here_and_now, data, experiments, T=data['num_timeslots'], n_clusters=num_cluster)
#    results[key] = se_costs
#    with open(f'results_{key}.json', 'w') as f:
#        json.dump(make_json_serializable({key: se_costs}), f)
#
## -------- SE_EV Policy --------
#print("------------------------------------------------> Running experiments: SE_EV")
#se_ev_costs = evaluate_policy_over_experiments(
#    EV_stochastic_optimization_here_and_now, data, experiments, T=data['num_timeslots'], n_clusters=1)
#results["SE_EV"] = se_ev_costs
#with open('results_SE_EV.json', 'w') as f:
#    json.dump(make_json_serializable({"SE_EV": se_ev_costs}), f)

# -------- ADP Policy --------
print("------------------------------------------------> Running experiments: ADP")
theta = train_value_function(data, gamma=0.95, I=50, K=5)
adp_costs = evaluate_policy_over_experiments(adp_policy_wrapper(theta), data, experiments, T=data['num_timeslots'])
results["ADP"] = adp_costs
with open('results_ADP.json', 'w') as f:
    json.dump(make_json_serializable({"ADP": adp_costs}), f)

# -------- Save All Results Summary --------
#with open('results_all.json', 'w') as f:
#    json.dump(make_json_serializable(results), f)
#

# -------- Plotting --------
def plot_cost_histograms(results):
    num_policies = len(results)
    fig, axes = plt.subplots(1, num_policies, figsize=(5 * num_policies, 5), sharey=True)
    if num_policies == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10").colors

    for ax, (name, costs), color in zip(axes, results.items(), colors):
        valid_costs = [c for c in costs if not np.isnan(c) and not np.isinf(c)]
        avg_cost = np.mean(valid_costs) if valid_costs else float('nan')

        ax.hist(valid_costs, bins=8, color=color, edgecolor='black', alpha=0.75)
        ax.set_title(f"{name}\nAvg Cost: {avg_cost:.1f}")
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Cost Distribution per Policy", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_line_comparison(results):
    colors = plt.get_cmap("tab10").colors
    num_experiments = len(next(iter(results.values())))
    x = range(num_experiments)

    plt.figure(figsize=(12, 6))
    for i, (name, costs) in enumerate(results.items()):
        clean_costs = [np.nan if np.isnan(c) or np.isinf(c) else c for c in costs]
        plt.plot(x, clean_costs, label=name, color=colors[i % len(colors)], marker='o')

    plt.title("Cost per Experiment Across Policies")
    plt.xlabel("Experiment")
    plt.ylabel("Total Cost")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------- Plotting Section --------
plot_cost_histograms(results)
plot_line_comparison(results)
