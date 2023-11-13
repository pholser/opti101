from gurobipy import GRB
import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd


batteries = ['Battery0', 'Battery1']
path = 'https://raw.githubusercontent.com/Gurobi/modeling-examples/master/optimization101/Modeling_Session_2/'
solar_values_read = pd.read_csv(path + 'pred_solar_values.csv')
time_periods = range(len(solar_values_read))
capacity = {'Battery0': 60, 'Battery1': 80}   # kW
p_loss = {'Battery0': 0.95, 'Battery1': 0.9}  # proportion
initial = {'Battery0': 0, 'Battery1': 0}      # kW
solar_values = round(solar_values_read.yhat, 3)
solar_values.reset_index(drop=True, inplace=True)
schedule = pd.read_csv(path + 'schedule_demand.csv')
avg_building = pd.read_csv(path + 'building_demand.csv')
total_demand = schedule.sched_demand + avg_building.build_demand
print(f"Total solar generation: {solar_values.sum()}\nTotal demand: {total_demand.sum()}")
avg_price = pd.read_csv(path + 'expected_price.csv')

m = gp.Model()
flow_in = m.addVars(batteries, time_periods, name='flow_in')
flow_out = m.addVars(batteries, time_periods, name='flow_out')
grid = m.addVars(time_periods, name='grid')
state = m.addVars(batteries, time_periods, name='state')
gen = m.addVars(time_periods, name='gen')
zwitch = m.addVars(batteries, time_periods, vtype=GRB.BINARY, name='zwitch')

m.addConstrs(
    (gp.quicksum(flow_out[b, t] - (p_loss[b] * flow_in[b, t]) for b in batteries)
        + gen[t]
        + grid[t]
        == total_demand[t]
        for t in time_periods),
    name='power_balance'
)
m.addConstrs(
    (state[b, 0] == initial[b] + (p_loss[b] * flow_in[b, 0]) - flow_out[b, 0]
        for b in batteries),
    name='initial_state'
)
m.addConstrs(
    (state[b, t] == state[b, t - 1] + (p_loss[b] * flow_in[b, t]) - flow_out[b, t]
        for b in batteries
        for t in time_periods
        if t > 0),
    name='subsequent_state'
)
m.addConstrs(
    (flow_in['Battery0', t] + flow_in['Battery1', t] + gen[t]
        <= solar_values[t]
        for t in time_periods),
    name='solar_avail'
)
m.addConstrs(
    (flow_in[b, t] <= 20 * zwitch[b, t]
        for b in batteries
        for t in time_periods),
    name='to_charge'
)
m.addConstrs(
    (flow_out[b, t] <= 20 * (1 - zwitch[b, t])
        for b in batteries
        for t in time_periods),
    name='or_not_to_charge'
)
for b, t in state:
    state[b, t].UB = capacity[b]

m.setObjective(
    gp.quicksum(avg_price.price[t] * grid[t] for t in time_periods),
    GRB.MINIMIZE
)

m.optimize()

print(f"Total cost of energy purchased from grid: {round(m.objVal, 2)}")

soln_in = pd.Series(m.getAttr('X', flow_in))
soln_out = pd.Series(m.getAttr('X', flow_out))
soln_level = pd.Series(m.getAttr('X', state))

print(f"Periods Battery 0 charges: {sum(soln_in['Battery0'] > 0)}")
print(f"Periods Battery 1 charges: {sum(soln_in['Battery1'] > 0)}")
print(f"Periods Battery 0 discharges: {sum(soln_out['Battery0'] > 0)}")
print(f"Periods Battery 1 discharges: {sum(soln_out['Battery1'] > 0)}")

plt.figure(figsize=(12, 5))
s0, = plt.plot(soln_level['Battery0'], c='orange')
s1, = plt.plot(soln_level['Battery1'], c='blue')
plt.ylabel('Battery State (kWh)')
plt.xlabel('Time Period')
plt.legend([s0, s1], ['Battery0', 'Battery1'])
plt.axhline(y=capacity['Battery0'], c='orange', linestyle='--', alpha=0.5)
plt.axhline(y=capacity['Battery1'], c='blue', linestyle='--', alpha=0.5)
print(f"Periods at Battery0 at Full Capacity: {sum(soln_level['Battery0'] == capacity['Battery0'])}")
print(f"Periods at Battery1 at Full Capacity: {sum(soln_level['Battery1'] == capacity['Battery1'])}")
