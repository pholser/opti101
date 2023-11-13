from gurobipy import GRB
import gurobipy as gp
import pandas as pd


production = {'Baltimore', 'Cleveland', 'Little Rock', 'Birmingham', 'Charleston'}
distribution = {'Columbia', 'Indianapolis', 'Lexington', 'Nashville', 'Richmond', 'St. Louis'}

m = gp.Model('widgets')

path = 'https://raw.githubusercontent.com/Gurobi/modeling-examples/master/optimization101/Modeling_Session_1/'

# squeeze=True makes the costs a series
transp_cost = pd.read_csv(path + 'cost.csv', index_col=[0, 1]).squeeze()

# pivot to view costs more easily
transp_cost.reset_index().pivot(index='production', columns='distribution', values='cost')

max_prod = pd.Series([180, 200, 140, 80, 180], index=production, name='max_production')
n_demand = pd.Series([89, 95, 121, 101, 116, 181], index=distribution, name='demand')
# max_prod.to_frame()
# n_demand.to_frame()

frac = 0.75

# x = {}
# for p in production:
#     for d in distribution:
#         x[p, d] = m.addVar(name=(p + '_to_' + d))
# m.update()

x = m.addVars(production, distribution, name='prod_ship')
m.update()

# x = m.addVars(transp_cost.index, name='prod_ship')
# m.update()

meet_demand = m.addConstrs(
    (gp.quicksum(x[p, d] for p in production) >= n_demand[d] for d in distribution),
    name='meet_demand'
)
m.update()
can_produce = m.addConstrs(
    (gp.quicksum(x[p, d] for d in distribution) <= max_prod[p] for p in production),
    name='can_produce'
)
m.update()
must_produce = m.addConstrs(
    (gp.quicksum(x[p, d] for d in distribution) >= frac * max_prod[p] for p in production),
    name='must_produce'
)
m.update()

m.setObjective(
    gp.quicksum(transp_cost[p, d] * x[p, d] for p in production for d in distribution),
    GRB.MINIMIZE
)

m.write('widget_shipment.lp')

m.optimize()

x_values = pd.Series(m.getAttr('X', x), name='shipment', index=transp_cost.index)
soln = pd.concat([transp_cost, x_values], axis=1)

all_vars = {v.varName: v.x for v in m.getVars()}

# Sum the shipment amount by production facility
ship_out = soln.groupby('production')['shipment'].sum()
print(ship_out)
print(pd.DataFrame({'Remaining': max_prod - ship_out, 'Utilization': ship_out / max_prod}))

print(pd.DataFrame(
    {'Remaining': [can_produce[p].Slack for p in production],
     'Utilization': [1 - can_produce[p].Slack / max_prod[p] for p in production]}))


m2 = gp.Model('widgets2')
x = m2.addVars(production, distribution, obj=transp_cost, name='prod_ship')
meet_demand = m2.addConstrs(
    (gp.quicksum(x[p, d] for p in production) >= n_demand[d] for d in distribution),
    name='meet_demand'
)
m2.update()
can_produce = m2.addConstrs(
    (gp.quicksum(x[p, d] for d in distribution) <= max_prod[p] for p in production if p != 'Birmingham'),
    name='can_produce'
)
m2.update()
must_produce = m2.addConstrs(
    (gp.quicksum(x[p, d] for d in distribution) >= frac * max_prod[p] for p in production if p != 'Birmingham'),
    name='must_produce'
)
m2.update()

xprod = m2.addVars(range(2), vtype=GRB.BINARY, obj=[50, 75], name='expand_birmingham_production')
m2.update()
bham_max_prod = m2.addConstr(
    gp.quicksum(x['Birmingham', d] for d in distribution) <= max_prod['Birmingham'] + 25*xprod[0] + 50*xprod[1],
    name='bham_max_prod'
)
m2.update()
bham_min_prod = m2.addConstr(
    gp.quicksum(x['Birmingham', d] for d in distribution) >= frac * (max_prod['Birmingham'] + 25*xprod[0] + 50*xprod[1]),
    name='bham_min_prod'
)
bham_choice = m2.addConstr(gp.quicksum(xprod[i] for i in range(2)) <= 1, name='bham_choice')
m2.update()

m2.optimize()
