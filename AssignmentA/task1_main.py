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
from task1_policy import make_dummy_decision
from task1_nextstate import nextstate
from task0 import simulate_wind_and_price
from task2 import *

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

def evaluate_policy_over_experiments(policy_func, data, experiments, T, n_clusters=None, verbose=False):
    n_experiments = len(experiments)
    n_timeslots = T
    policy_costs = np.full(n_experiments, np.nan)

    for e, (wind, price) in enumerate(experiments):
        hydrogen_stock = {}
        ele = {}
        hydrogen_stock[0] = 0
        ele[0] = 0
        total_cost = 0
        feasible = True

        for t in range(n_timeslots):
            # Decision from the policy
            if policy_func == stochastic_optimization_here_and_now:
                y_on, y_off, P2H, H2P, p = policy_func(data, wind, price, hydrogen_stock, ele, t, n_timeslots, n_clusters)
            else:
                y_on, y_off, P2H, H2P, p = policy_func(data, wind, price, hydrogen_stock, ele, t, n_timeslots)
            # Check feasibility
            feasible, y_on, y_off, P2H, H2P, p = check_feasibility(
                data, y_on, y_off, P2H, H2P, p, wind, hydrogen_stock, ele, t
            )

            if not feasible:
                if verbose:
                    print(f"[Experiment {e} | Time {t}] ❌ Infeasible")
                    print(f"  y_on: {y_on}, y_off: {y_off}, P2H: {P2H}, H2P: {H2P}, p: {p}, wind: {wind[t]}, price: {price[t]}")
                    print("demand: ", data['demand_schedule'][t])
                break

            # Update cost
            total_cost += p * price[t] + ele[t] * data['electrolyzer_cost']

            # Update state
            hydrogen_stock[t + 1], ele[t + 1] = nextstate(data, hydrogen_stock[t], ele[t], P2H, H2P, y_on, y_off)

        if feasible:
            policy_costs[e] = total_cost

    return policy_costs


# try with dummy policy
if __name__ == "__main__":
    #generate experiments
    experiments = generate_experiments(data['num_timeslots'], n=20)

    #evaluate policy
    results = evaluate_policy_over_experiments(make_dummy_decision, data, experiments, data['num_timeslots'], verbose=True)
    print("Dummy policy costs: ", results)

