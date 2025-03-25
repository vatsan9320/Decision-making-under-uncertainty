# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:04:55 2025

@author: geots
"""

import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from sklearn.cluster import KMeans

def make_stochastic_here_and_now_decision(p1, N):
    from v2_02435_two_stage_problem_data import load_the_data
    from v2_price_process import sample_next
    
    number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()
    
    it = 100
    p_scen = np.zeros((len(W), it))
    for i, w in enumerate(W):
        for j in range(it):
            p_scen[i, j] = sample_next(p1[w])
    
    prob = 1/100
    
    model = ConcreteModel()
    
    model.W = Set(initialize=W)
    model.S = RangeSet(N)
    
    model.x1 = Var(model.W, within=NonNegativeReals) # quantity order in stage 1
    model.x2 = Var(model.W, model.S, within=NonNegativeReals) # quantity order in stage 2
    model.z1 = Var(model.W, within=NonNegativeReals) # warehouse storage in stage 1
    model.z2 = Var(model.W, model.S, within=NonNegativeReals) # warehouse storage in stage 2
    model.m1 = Var(model.W, within=NonNegativeReals) # missing quantity in stage 1
    model.m2 = Var(model.W, model.S, within=NonNegativeReals) # missing quantity in stage 2
    model.ys1 = Var(model.W, model.W, within=NonNegativeReals) # quantity send in stage 1
    model.ys2 = Var(model.W, model.W, model.S, within=NonNegativeReals)  # quantity send in stage 2
    model.yr1 = Var(model.W, model.W, within=NonNegativeReals) # quantity received in stage 1
    model.yr2 = Var(model.W, model.W, model.S, within=NonNegativeReals)  # quantity received in stage 2
    
    model.obj = Objective(
        expr=
        sum(model.x1[w] * p1[w] for w in W) +
        sum(model.x2[w, s] * p_scen[i, s] * prob[s] for i, w in enumerate(W) for s in model.S) +
        sum(model.ys1[w, q] * cost_tr[w, q] for w in W for q in W) +
        sum(model.ys2[w, q, s] * cost_tr[w, q] * prob[s] for w in W for q in W for s in model.S) +
        sum(model.m1[w] * cost_miss[w] for w in W) +
        sum(model.m2[w, s] * cost_miss[w] * prob[s] for w in W for s in model.S),
        sense=minimize
    )
    
    # STAGE 1
    #Constraint on transport capacity
    model.TrCap1 = ConstraintList()
    for w in W:
        for q in W:
            model.TrCap1.add(model.ys1[w, q] <= transport_capacities[w, q])
    
    #Constraint on quantity sent equal to quantity received
    model.QTrans1 = ConstraintList()
    for w in W:
        for q in W:
            model.QTrans1.add(model.ys1[w, q] == model.yr1[q, w])
    
    #Constraint on storage capacity
    model.StCap1 = ConstraintList()
    for w in W:
        model.StCap1.add(model.z1[w] <= warehouse_capacities[w])
    
    #Constraint on demand fulfillment
    model.Demand1 = ConstraintList()
    for w in W:
        model.Demand1.add(model.x1[w] + model.m1[w] + initial_stock[w] + sum(model.yr1[w, q] for q in W) ==
                          demand_trajectory[w, 1] + model.z1[w] + sum(model.ys1[w, q] for q in W))
    
    #Constraint on (stored) amount sent between warehouses
    model.YsCons1 = ConstraintList()
    for w in W:
        model.YsCons1.add(sum(model.ys1[w, q] for q in W) <= initial_stock[w])
        
        
    ## Stage 2 Constraints
    model.TrCap2 = Constraint(model.W, model.W, model.S, rule=lambda m, w, q, s: m.ys2[w, q, s] <= m.Ct[w, q])
    model.QTrans2 = Constraint(model.W, model.W, model.S, rule=lambda m, w, q, s: m.ys2[w, q, s] == m.yr2[q, w, s])
    model.StCap2 = Constraint(model.W, model.S, rule=lambda m, w, s: m.z2[w, s] <= m.Cs[w])
    model.Demand2 = Constraint(model.W, model.S, rule=lambda m, w, s: m.x2[w, s] + m.m2[w, s] + m.z1[w] + sum(m.yr2[w, q, s] for q in W) == m.D[w, 2] + m.z2[w, s] + sum(m.ys2[w, q, s] for q in W))
    model.YsCons2 = Constraint(model.W, model.S, rule=lambda m, w, s: sum(m.ys2[w, q, s] for q in W) <= m.z1[w])
    
        
    
    
    opt = SolverFactory('gurobi')
    opt.solve(model)
    
    return p2, model.x1, model.z1, model.m1, model.ys1, model.yr1, model.obj.expr.value