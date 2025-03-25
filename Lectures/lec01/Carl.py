# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:30:51 2025

@author: geots
"""

from pyomo.environ import *

# Create a model
model = ConcreteModel()

# Declare variables
model.xC = Var(bounds=(0, 50), within=Integers)
model.xS = Var(bounds=(0, 200), within=Integers)
model.yS = Var(bounds=(0, 1), within=Binary)
model.yC = Var(bounds=(0, 1), within=Binary)

# Objective function: Maximization of profits
model.profit = Objective(
    expr=(400 - 200) * model.xC + (70 - 30) * model.xS - 1000 * model.yC,
    sense=maximize
)

# Constraints

#Constraint on available acres
model.Acres = Constraint(expr=model.xC + 0.2 * model.xS <= 72)

#Constraint on maximum working hours
model.WorkingHours = Constraint(expr=150 * model.xC + 25 * model.xS <= 10000)

#Minimum of 100 sheep constraint
model.AtLeast100Sheep1 = Constraint(expr=model.xS - 100 * model.yS >= 0)
model.AtLeast100Sheep2 = Constraint(expr=model.xS - 200 * model.yS <= 0)

#Maximum of 10 cows without milk machine
model.MilkMachine1 = Constraint(expr=model.xC - 10 - 40 * model.yC <= 0)
model.MilkMachine2 = Constraint(expr=model.xC - 11 * model.yC >= 0)

# Create a solver
solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

# Solve the model
results = solver.solve(model, tee=True)

# Check if an optimal solution was found
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found")

    # Print out variable values and objective value
    print("Variable values:")
    print(f"xC: {value(model.xC):.3f}")
    print(f"xS: {value(model.xS):.3f}")
    print(f"yS: {value(model.yS):.3f}")
    print(f"yC: {value(model.yC):.3f}")
    print(f"\nObjective value: {value(model.profit):.3f}\n")
else:
    print("No optimal solution found.")