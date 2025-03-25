# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:59:29 2024

@author: geots
"""

import matplotlib.pyplot as plt

initial_state = {'hydrogen': 0, 'electrolyzer_status': 0}

def plot_results(data, times, wind_trajectory, demand_schedule, results, price_trajectory):
    # Plot results
    plt.figure(figsize=(14, 12))

    plt.subplot(9, 1, 1)
    plt.plot(times, wind_trajectory, label="Wind Power", color="blue")
    plt.ylabel("Wind Power")
    plt.legend()

    plt.subplot(9, 1, 2)
    plt.plot(times, demand_schedule, label="Demand Schedule", color="orange")
    plt.ylabel("Demand")
    plt.legend()

    plt.subplot(9, 1, 3)
    plt.step(times, results['electrolyzer_status'], 'o' ,label="Electrolyzer Status", color="red", where="post")
    plt.ylabel("El. Status")
    plt.legend()

    plt.subplot(9, 1, 4)
    plt.plot(times, results['hydrogen_storage_level'], label="Hydrogen Level", color="green")
    plt.ylabel("Hydr. Level")
    plt.legend()

    plt.subplot(9, 1, 5)
    plt.plot(times, results['power_to_hydrogen'], label="p2h", color="orange")
    plt.ylabel("p2h")
    plt.legend()

    plt.subplot(9, 1, 6)
    plt.plot(times, results['hydrogen_to_power'], label="h2p", color="blue")
    plt.ylabel("h2p")
    plt.legend()

    plt.subplot(9, 1, 7)
    plt.plot(times, results['grid_power'], label="Grid Power", color="green")
    plt.ylabel("Grid Power")
    plt.legend()

    plt.subplot(9, 1, 8)
    plt.plot(times, price_trajectory, label="price", color="red")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend()

    plt.subplot(9, 1, 9)
    plt.plot(times, [results['hydrogen_to_power'][t]*data['conversion_h2p'] + results['grid_power'][t] + wind_trajectory[t] - results['power_to_hydrogen'][t]  for t in range(len(times))], label="Total Power", color="blue")
    plt.plot(times, demand_schedule, label="Demand Schedule", color="orange")
    plt.ylabel("Power")
    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    plt.show()