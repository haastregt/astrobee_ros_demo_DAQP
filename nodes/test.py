from mpc import MPC
from astrobee import Astrobee
from set_operations import SetOperations
from simulation import EmbeddedSimEnvironment
import scipy
import numpy as np
import polytope as pc
from ctypes import *
from scipy.spatial.transform import Rotation

from MPCParams import MPCParams
from MPCSolver import MPCSolver

params = MPCParams()
ctl = MPCSolver(params)
state = np.zeros((12,1))
state[0:3] = 0.1
state[9] = 1
u_traj = ctl.solve(state)
print(u_traj)
'''
SET_TYPE = "LQR"  # Terminal invariant set type: select 'zero' or 'LQR'
MPC_HORIZON = 10
Q = np.eye(12)
R = np.eye(6) * 0.2
referenceTrajectory = np.zeros((12, 1))

# Get the system
honey = Astrobee()
A = honey.Ad
B = honey.Bd

# Solve the ARE for our system to extract the terminal weight matrix P
P_LQR = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

# Instantiate limits
u_lim = np.array([[0.85, 0.41, 0.41, 0.085, 0.041, 0.041]]).T
x_lim = 100*np.array([[1.2, 0.1, 0.1,
                   0.5, 0.5, 0.5,
                   0.2, 0.2, 0.2,
                   0.1, 0.1, 0.1]]).T

# Translation Dynamics
At = A[0:6, 0:6].reshape((6, 6))
Bt = B[0:6, 0:3].reshape((6, 3))
Qt = Q[0:6, 0:6].reshape((6, 6))
Rt = R[0:3, 0:3].reshape((3, 3))
x_lim_t = x_lim[0:6, :].reshape((6, 1))
u_lim_t = u_lim[0:3, :].reshape((3, 1))
set_ops_t = SetOperations(At, Bt, Qt, Rt, xlb=-x_lim_t, xub=x_lim_t)

# Attitude Dynamics
Aa = A[6:, 6:].reshape((6, 6))
Ba = B[6:, 3:].reshape((6, 3))
Qa = Q[6:, 6:].reshape((6, 6))
Ra = R[3:, 3:].reshape((3, 3))
x_lim_a = x_lim[6:, :].reshape((6, 1))
u_lim_a = u_lim[3:, :].reshape((3, 1))
set_ops_a = SetOperations(Aa, Ba, Qa, Ra, xlb=-x_lim_a, xub=x_lim_a)

if SET_TYPE == "zero":
    Xf_t = set_ops_t.zeroSet()
    Xf_a = set_ops_a.zeroSet()
elif SET_TYPE == "LQR":
    # Create constraint polytope for translation and attitude
    Cub = np.eye(3)
    Clb = -1 * np.eye(3)

    Cb_t = np.concatenate((u_lim_t, u_lim_t), axis=0)
    C_t = np.concatenate((Cub, Clb), axis=0)

    Cb_a = np.concatenate((u_lim_a, u_lim_a), axis=0)
    C_a = np.concatenate((Cub, Clb), axis=0)

    Ct = pc.Polytope(C_t, Cb_t)
    Ca = pc.Polytope(C_a, Cb_a)

    # Get the LQR set for each of these
    Xf_t = set_ops_t.LQRSet(Ct)
    Xf_a = set_ops_a.LQRSet(Ca)
else:
    print("Wrong choice of SET_TYPE, select 'zero' or 'LQR'.")

Xf = pc.Polytope(scipy.linalg.block_diag(Xf_t.A, Xf_a.A), np.concatenate((Xf_t.b, Xf_a.b), axis=0))

# Create MPC controller
ctl = MPC(model=honey, Q=Q, R=R, P=P_LQR, N=MPC_HORIZON,
          ulb=-u_lim, uub=u_lim,
          xlb=-x_lim, xub=x_lim, Xf=Xf)
ctl.SetReference(referenceTrajectory)

state = np.zeros((13,1))
state[0:3] = 0.1
state[9] = 1
#u_traj = ctl.GetControl(state)
#print(u_traj)

sim_env = EmbeddedSimEnvironment(model=honey,
                                 dynamics=honey.LinearizedDiscreteDynamics,
                                 controller=ctl.GetControlSimulation,
                                 time=20)

state = np.zeros((12,1))
#state[0:3] = 0
#state[3:6] = 0
#state[8] = 0.349
t, y, u = sim_env.run(state)
sim_env.visualize()
'''

'''
import daqp
import numpy as np
from ctypes import * 
import ctypes.util

# Define the problem
H = np.array([[1, 0], [0, 1]],dtype=c_double)
f = np.array([1, 1],dtype=c_double)
A = np.array([[1, 1], [1, -1]],dtype=c_double)
bupper = np.array([1,2,3,4],dtype=c_double)
blower = np.array([-1,-2,-3,-4],dtype=c_double)
sense = np.array([0,0,0,0],dtype=c_int)

d = daqp.daqp()
print("Starting DAQP solving")
(xstar,fval,exitflag,info) = d.quadprog(H,f,A,bupper,blower,sense)
print(xstar)
'''