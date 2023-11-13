from gurobipy import GRB
import datetime
import gurobipy as gp
import pandas as pd


path = 'https://raw.githubusercontent.com/Gurobi/modeling-examples/master/optimization101/bike_share/'
stations = pd.read_csv(path + 'top_stations.csv', index_col='station')
stations_flow = pd.read_csv(path + 'stations_flow.csv')
stations_flow['datetime'] = stations_flow['datetime'].map(pd.to_datetime)

pd.options.mode.chained_assignment = None
morning_flow = stations_flow[stations_flow['datetime'].dt.hour.between(7, 9)]
morning_flow['date'] = morning_flow['datetime'].dt.date
morning_flow['time'] = morning_flow['datetime'].dt.hour
flow_df = morning_flow.loc[morning_flow['date'] == datetime.date(2022, 8, 1)]
flow_df.set_index(['station', 'time'], inplace=True)

# N: number of bikes on hand that we can assign to stations at a given hour
num_bikes = 25

station_names = list(stations.index)
time_rng = morning_flow['time'].drop_duplicates().values
station_time = flow_df.index   # pairs of (station, time)
start_forecast = flow_df.start_forecast
end_forecast = flow_df.end_forecast
capacity = stations.capacity

m = gp.Model('bike_rebalancing')
y = m.addVars(station_time, lb=0, vtype=GRB.CONTINUOUS, name='add_bikes_to_station_at_time')
z = m.addVars(station_time, lb=0, vtype=GRB.CONTINUOUS, name='remove_bikes_from_station_at_time')
ell = m.addVars(station_time, lb=0, vtype=GRB.CONTINUOUS, name='loss')

m.addConstrs(
    (y[i, t] <= capacity[i] for i, t in station_time),
    name='cannot_add_more_bikes_than_capacity'
)
m.addConstrs(
    (z[i, t] <= max(0, end_forecast[i, t] - start_forecast[i, t])
     for i, t in station_time),
    name='cannot_remove_more_than_you_end_up_with'
)
m.addConstrs(
    (z[i, t] >= end_forecast[i, t] - start_forecast[i, t] - capacity[i]
     for i, t in station_time),
    name='cannot_remove_less_than'
)
q = m.addVars(station_time, lb=0, vtype=GRB.CONTINUOUS, name='inventory')
t0 = 7
m.addConstrs(
    (q[i, t0] == 0 for i in stations.index),
    name='initial_inventory'
)
aux = m.addVars(station_time, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux')
m.addConstrs(
    (aux[i, t] ==
     end_forecast[i, t - 1]
        + y[i, t - 1]
        + q[i, t - 1]
        - start_forecast[i, t - 1]
        - z[i, t - 1])
    for i, t in station_time if t != t0
)
m.addConstrs(
    (q[i, t] == gp.max_(0, aux[i, t]) for i, t in station_time if t != t0),
    name='inventory_constraints'
)
m.addConstrs(
    (q[i, t] <= capacity[i] for i, t in station_time),
    name='inventory_upper_bound'
)
loss_aux = m.addVars(station_time, lb=0, vtype=GRB.CONTINUOUS, name='loss_aux')
m.addConstrs(
    (loss_aux[i, t] ==
     start_forecast[i, t]
     + z[i, t]
     - end_forecast[i, t]
     - y[i, t]
     - q[i, t]
     for i, t in station_time),
    name='defn_of_loss'
)
loss = m.addVars(station_time, lb=0, vtype=GRB.CONTINUOUS, name='loss')
m.addConstrs(
    (loss[i, t] == gp.max_(0, loss_aux[i, t]) for i, t in station_time),
    name='loss_non_negative'
)
m.addConstrs(
    (y.sum('*', t) <= num_bikes for t in time_rng),
    name='limit_on_number_of_bikes_added'
)

m.setObjective(loss.sum(), GRB.MINIMIZE)

m.optimize()
print(m.getAttr('X', y))
print(m.getAttr('X', z))
