import numpy as np
import polytope as pc
import scipy
from astrobee import Astrobee
from set_operations import SetOperations

class MPCParams():
    def __init__(self):
        '''
        Here are all tuneable parameters!
        '''
        self.method = "cvxpygen"    # "manual", "cvxpygen", "acados"
        self.solver = "OSQP"        # "OSQP", "DAQP"
        self.qp_format = "dense"   # "sparse", "dense"

        self.MPC_HORIZON = 10  # Horizon length  
        self.SET_TYPE = "LQR"  # Terminal invariant set type: select 'zero' or 'LQR'
        
        # Get the system (can be initialised with mass, inertia, and sampling time)
        self.model = Astrobee() 

        # State and Control cost matrices
        self.Q = np.eye(12) * 1
        self.R = np.eye(6) * 10 #10
        self.R[3,3] = 100 #100
        self.R[4,4] = 100 #100
        self.R[5,5] = 100 #100

        # Control and state bounds
        self.u_lim = np.array([[0.85, 0.41, 0.41, 0.085, 0.041, 0.041]]).T
        self.x_lim = 100*np.array([[1.2, 0.1, 0.1,
                                    0.5, 0.5, 0.5,
                                    0.2, 0.2, 0.2,
                                    0.1, 0.1, 0.1]]).T

        # Reference
        self.referenceTrajectory = np.zeros((12, self.MPC_HORIZON))

        self.ComputeTerminalSet()

    def ComputeTerminalSet(self):
        A = self.model.Ad
        B = self.model.Bd

        # Solve the ARE for our system to extract the terminal weight matrix P
        self.P_LQR = np.matrix(scipy.linalg.solve_discrete_are(A, B, self.Q, self.R))

        # Translation Dynamics
        At = A[0:6, 0:6].reshape((6, 6))
        Bt = B[0:6, 0:3].reshape((6, 3))
        Qt = self.Q[0:6, 0:6].reshape((6, 6))
        Rt = self.R[0:3, 0:3].reshape((3, 3))
        x_lim_t = self.x_lim[0:6, :].reshape((6, 1))
        u_lim_t = self.u_lim[0:3, :].reshape((3, 1))
        set_ops_t = SetOperations(At, Bt, Qt, Rt, xlb=-x_lim_t, xub=x_lim_t)

        # Attitude Dynamics
        Aa = A[6:, 6:].reshape((6, 6))
        Ba = B[6:, 3:].reshape((6, 3))
        Qa = self.Q[6:, 6:].reshape((6, 6))
        Ra = self.R[3:, 3:].reshape((3, 3))
        x_lim_a = self.x_lim[6:, :].reshape((6, 1))
        u_lim_a = self.u_lim[3:, :].reshape((3, 1))
        set_ops_a = SetOperations(Aa, Ba, Qa, Ra, xlb=-x_lim_a, xub=x_lim_a)

        if self.SET_TYPE == "zero":
            Xf_t = set_ops_t.zeroSet()
            Xf_a = set_ops_a.zeroSet()
        elif self.SET_TYPE == "LQR":
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

        self.Xf = pc.Polytope(scipy.linalg.block_diag(Xf_t.A, Xf_a.A), np.concatenate((Xf_t.b, Xf_a.b), axis=0))
