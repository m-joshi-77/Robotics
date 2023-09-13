# Importing the required packages:
import numpy as np
import csv
import modern_robotics as mr
import pandas as pd


M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]
Mlist = [M01, M12, M23, M34, M45, M56, M67]
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]

# Gravity vector:
g = np.array([0, 0, -9.81])

# Ftip (Spatial force applied by the end-effector):
Ftip = np.ones(6)*0

# Joint torque list
taulist = ([0, 0, 0, 0, 0, 0])

# Code for simulation 1:

thetalist = ([0, 0, 0, 0, 0, 0])
dthetalist = np.zeros(6)
simulation1 = np.empty([0,6])
step = 0.01
for i in range(300):
    ddthetalist = mr.ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist)
    thetalist, dthetalist = mr.EulerStep(thetalist, dthetalist, ddthetalist, step)
    simulation1 = np.concatenate([simulation1, thetalist.reshape([1,6])], axis=0)

# Populating the simulation1.csv file:
pd.DataFrame(simulation1).to_csv("simulation1.csv", index=False, header=False)

# Code for simulation 2:

thetalist = ([0, -1, 0, 0, 0, 0])
dthetalist = np.zeros(6)
simulation2 = np.empty([0,6])
step = 0.01
for i in range(500):
    ddthetalist = mr.ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist)
    thetalist, dthetalist = mr.EulerStep(thetalist, dthetalist, ddthetalist, step)
    simulation2 = np.concatenate([simulation2, thetalist.reshape([1,6])], axis=0)

# Populating the simulation2.csv file:
pd.DataFrame(simulation2).to_csv("simulation2.csv", index=False, header=False)