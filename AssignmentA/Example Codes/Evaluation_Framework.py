# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:48:47 2025

@author: geots
"""

import numpy as np

# This code applies and evaluates a policy over 40 different experiments and gives you the average policy's cost


from my_policy import make_here_and_now_decision
from dummy_policy import make_dummy_decision
from problem_data import load_the_data
from simulation_experiments import simulation_experiments_creation
from feasibility_check import check_feasibility

# Loading the problem's parameters
(
    number_of_warehouses,
    W,
    cost_miss,
    cost_tr,
    warehouse_capacities,
    transport_capacities,
    initial_stock,
    number_of_sim_periods,
    sim_T,
    demand_trajectory,
) = load_the_data()

# Creating the random experiments on which the policy will be evaluated
number_of_experiments, Expers, Price_experiments = simulation_experiments_creation(
    number_of_warehouses, W, number_of_sim_periods
)

# Initialization of the decision variables and policy cost
x = {}
send = {}
receive = {}
z = {}
m = {}
policy_cost = np.full((number_of_experiments, number_of_sim_periods), 99999999)
policy_cost_at_experiment = np.full(number_of_experiments, 99999999)

# for each experiment
for e in Expers:
    # and for each timeslot of the horizon
    for tau in sim_T:
        # Set each warehouse's stock level
        current_stock = initial_stock if tau == 1 else z[(e, tau - 1)]

        # Observe current demands and prices
        current_demands = demand_trajectory[:, tau - 1]
        current_prices = Price_experiments[e, :, tau - 1]

        # Call policy to make a decision for here and now
        (
            x[(e, tau)],
            send[(e, tau)],
            receive[(e, tau)],
            z[(e, tau)],
            m[(e, tau)],
        ) = make_here_and_now_decision(
            number_of_sim_periods, tau, current_stock, current_prices
        )

        # Check whether the policy's here-and-now decisions are feasible/meaningful
        successful = check_feasibility(
            x[(e, tau)],
            send[(e, tau)],
            receive[(e, tau)],
            z[(e, tau)],
            m[(e, tau)],
            current_stock,
            current_demands,
            warehouse_capacities,
            transport_capacities,
        )

        # If not, then the policy's decisions are discarded for this timeslot, and the dummy policy is used instead
        if not successful:
            print("DECISION DOES NOT MEET THE CONSTRAINTS FOR THIS TIMESLOT. THE DUMMY POLICY WILL BE USED INSTEAD")
            print(
                e,
                number_of_sim_periods,
                tau,
                current_stock,
                current_demands,
                x[(e, tau)],
                send[(e, tau)],
                receive[(e, tau)],
                z[(e, tau)],
                m[(e, tau)],
            )
            x[(e, tau)], send[(e, tau)], receive[(e, tau)], z[(e, tau)], m[(e, tau)] = make_dummy_decision(
                number_of_sim_periods, tau, current_stock, current_prices
            )

        policy_cost[e, tau - 1] = sum(
            current_prices[w] * x[(e, tau)][w]
            + cost_miss[w] * m[(e, tau)][w]
            + sum(cost_tr[w, q] * receive[(e, tau)][w, q] for q in W)
            for w in W
        )

    policy_cost_at_experiment[e] = sum(policy_cost[e, tau - 1] for tau in sim_T)

FINAL_POLICY_COST = sum(policy_cost_at_experiment[e] for e in Expers) / number_of_experiments
print("THE FINAL POLICY EXPECTED COST IS", FINAL_POLICY_COST)