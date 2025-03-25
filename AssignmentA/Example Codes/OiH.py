# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:58:17 2025

@author: geots
"""

from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

def Calculate_OiH_solution(p1, p2):
    from V2_02435_two_stage_problem_data import load_the_data
    from V2_price_process import *

    number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()

    # Declare model
    model = ConcreteModel()

    # Sets and Parameters
    model.W = Set(initialize=W)
    model.T = Set(initialize=sim_T)
    model.b = Param(model.W, initialize=cost_miss)
    model.e = Param(model.W * model.W, initialize=cost_tr)
    model.Cs = Param(model.W, initialize=warehouse_capacities)
    model.Ct = Param(model.W * model.W, initialize=transport_capacities)
    model.D = Param(model.W, model.T, initialize=demand_trajectory)
    model.z0 = Param(model.W, initialize=initial_stock)

    # Variables
    
    # quantity order
    model.x = Var(model.W, model.T, domain=NonNegativeReals)
    
    # warehouse storage
    model.z = Var(model.W, model.T, domain=NonNegativeReals)
    
    # missing quantity
    model.m = Var(model.W, model.T, domain=NonNegativeReals)
    
    # quantity send
    model.ys = Var(model.W, model.W, model.T, domain=NonNegativeReals)
    
    # quantity received
    model.yr = Var(model.W, model.W, model.T, domain=NonNegativeReals)

    # Objective Function
    def objective_rule(model):
        return sum(model.x[w,1] * p1[w] + model.x[w,2] * p2[w] for w in model.W) + \
               sum(model.ys[w,q,t] * model.e[w,q] for w in model.W for q in model.W for t in model.T) + \
               sum(model.m[w,t] * model.b[w] for w in model.W for t in model.T)
    
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    
    #Constraint on transport capacity
    model.TrCap = Constraint(model.W, model.W, model.T, rule=lambda model, w, q, t: model.ys[w,q,t] <= model.Ct[w,q])
    
    #Constraint on quantity sent equal to quantity received
    model.QTrans = Constraint(model.W, model.W, model.T, rule=lambda model, w, q, t: model.ys[w,q,t] == model.yr[q,w,t])
   
    #Constraint on storage capacity
    model.StCap = Constraint(model.W, model.T, rule=lambda model, w, t: model.z[w,t] <= model.Cs[w])
    
    #Constraint on coffee balance
    model.Demand = Constraint(model.W, model.T, rule=lambda model, w, t: 
        model.x[w,t] + model.m[w,t] + (model.z0[w] if t==1 else model.z[w,t-1]) + \
        sum(model.yr[w,q,t] for q in model.W) == model.D[w,t] + model.z[w,t] + sum(model.ys[w,q,t] for q in model.W))
    
     #Constraint on (stored) amount sent between warehouses    
    model.YsCons = Constraint(model.W, model.T, rule=lambda model, w, t: 
        sum(model.ys[w,q,t] for q in model.W) <= (model.z0[w] if t==1 else model.z[w,t-1]))

    # Solve model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model.x, model.z, model.m, model.ys, model.yr, model.obj.expr
