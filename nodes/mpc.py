import numpy as np
import scipy
import daqp
import osqp
from ctypes import * 
import ctypes.util

import time

from scipy.spatial.transform import Rotation


class MPC():
    def __init__(self, model, N, Q, P, R,
                 ulb=None, uub=None, xlb=None, xub=None, Xf=None):

        self.USE_DAQP = True
        self.CONDENSED_MODEL = True

        self.FIRST_ITERATION = True

        """
        Constructor for the MPC class.

        :param model: System model
        :type model: Astrobee
        :param N: horizion length
        :type N: int
        :param Q: state weight matrix, defaults to None
        :type Q: np.ndarray
        :param P: terminal state weight matrix, defaults to None
        :type P: np.ndarray
        :param R: control weight matrix, defaults to None
        :type R: np.ndarray
        :param ulb: control lower bound vector, defaults to None
        :type ulb: np.ndarray, optional
        :param uub: control upper bound vector, defaults to None
        :type uub: np.ndarray, optional
        :param xlb: state lower bound vector, defaults to None
        :type xlb: np.ndarray, optional
        :param xub: state upper bound vector, defaults to None
        :type xub: [type], optional
        :param terminal_constraint: terminal constriant polytope, defaults to None
        :type terminal_constraint: Polytope, optional

        Input is in the MPC form:
            min u       0.5*x_N^T*Q_f*x_N _ 0.5*sum(x_k^T*Q*z_k + u_k^T*R*u_k)
            subject to  x_k+1 = Ad*x_k + Bd*u_k
                        xlb <= x_k <= xub
                        ulb <= u_k <= uub
                        x_N c X_f
        
        We need to get this into the form:
            min u       0.5*u^T*H*u + f^T*u
            subject to  A*u <= b
        
        Where u is a vector of the control for each timestep
        """

        self.model = model
        self.dt = model.dt
        self.Nx, self.Nu = model.n, model.m
        self.Nt = N

        # TODO make sure there is some checking in case of un-supplied
        # optional arguments

        if self.CONDENSED_MODEL:
            self.Nconstraints = 2*self.Nt*(self.Nx + self.Nu) + Xf.A.shape[0]

            # Q = diag(Q, ..., Q, Q_f)
            self.Q = scipy.linalg.block_diag(*([Q] * (N)), P)
            # R = diag(R, ..., R)
            self.R = scipy.linalg.block_diag(*([R] * N))
            # No lower limit, so -inf for all constraint entries
            self.blower = np.full((self.Nconstraints,), -np.inf)
            # Sense is the type of constraints. Using 0, all inequalities
            self.sense = np.full((self.Nconstraints,), 0)

            # Stack one time for upper bounds, one time for lower bounds
            stateConstraints = np.vstack((np.eye(self.Nx*self.Nt),
                                        -np.eye(self.Nx*self.Nt)))
            # Append the final state constraints
            stateConstraints = np.block([[stateConstraints, np.zeros((2*self.Nx*(self.Nt), self.Nx))],
                                        [np.zeros((Xf.A.shape[0], self.Nx*(self.Nt))), Xf.A]])

            # Stack one time for upper bounds, one time for lower bounds
            controlConstraints = np.vstack((np.eye(self.Nu*self.Nt),
                                            -np.eye(self.Nu*self.Nt)))

            self.Az = np.vstack((stateConstraints, 
                                np.zeros((controlConstraints.shape[0], stateConstraints.shape[1]))))
            self.Au = np.vstack((np.zeros((stateConstraints.shape[0], controlConstraints.shape[1])), 
                                controlConstraints))

            # This represents the constraints |x| < b_x and |u| < b_u
            self.b = np.hstack((np.tile(xub.squeeze(), self.Nt),
                                np.tile(-xlb.squeeze(), self.Nt), 
                                Xf.b, 
                                np.tile(uub.squeeze(), self.Nt),
                                np.tile(-ulb.squeeze(), self.Nt)))
        else:
            self.Nequalities = self.Nx*self.Nt
            self.Ninequalities = 2*self.Nt*(self.Nx + self.Nu) + Xf.A.shape[0]
            self.Nconstraints = self.Nequalities + self.Ninequalities

            # Q = diag(Q, ..., Q, Q_f)
            self.Q = scipy.linalg.block_diag(*([Q] * (N)), P)
            # R = diag(R, ..., R)
            self.R = scipy.linalg.block_diag(*([R] * N))
            # No lower limit, so -inf for all constraint entries
            self.blower = np.full((self.Nconstraints,), -np.inf)
            # Sense is the type of constraints. Using 0, all inequalities
            senseEqual = np.full((self.Nequalities,), 5)
            senseInequal = np.full((self.Ninequalities,), 0)
            self.sens = np.vstack((senseEqual,
                                   senseInequal))

            # Stack one time for upper bounds, one time for lower bounds
            stateConstraints = np.vstack((np.eye(self.Nx*self.Nt),
                                        -np.eye(self.Nx*self.Nt)))
            # Append the final state constraints
            stateConstraints = np.block([[stateConstraints, np.zeros((2*self.Nx*(self.Nt), self.Nx))],
                                        [np.zeros((Xf.A.shape[0], self.Nx*(self.Nt))), Xf.A]])

            # Stack one time for upper bounds, one time for lower bounds
            controlConstraints = np.vstack((np.eye(self.Nu*self.Nt),
                                            -np.eye(self.Nu*self.Nt)))

            self.Az = np.vstack((stateConstraints, 
                                np.zeros((controlConstraints.shape[0], stateConstraints.shape[1]))))
            self.Au = np.vstack((np.zeros((stateConstraints.shape[0], controlConstraints.shape[1])), 
                                controlConstraints))

            # This represents the constraints |x| < b_x and |u| < b_u
            self.bIneq = np.hstack((np.tile(xub.squeeze(), self.Nt),
                                    np.tile(-xlb.squeeze(), self.Nt), 
                                    Xf.b, 
                                    np.tile(uub.squeeze(), self.Nt),
                                    np.tile(-ulb.squeeze(), self.Nt)))

        self.d = daqp.daqp()
        self.m = osqp.OSQP()

    def UpdateLinearization(self, linearizationPoint):
        # Recompute matrices for new linearization point
        # TODO: actually use linearizationPoint. For now just linearized around 0 (already done in astrobee class)
        Ad = self.model.Ad
        Bd = self.model.Bd

        if self.CONDENSED_MODEL:
            self.F = np.eye(self.Nx)
            self.G = np.zeros(((self.Nt+1)*self.Nx, self.Nt*self.Nu))
            for t in range(self.Nt):
                self.F = np.vstack((self.F, np.linalg.matrix_power(Ad, t+1)))
                GProduct = np.matmul(np.linalg.matrix_power(Ad, t), Bd)
                for i in range(self.Nt-t):
                    self.G[(t+i+1)*self.Nx:(t+i+2)*self.Nx, i*self.Nu:(i+1)*self.Nu] = GProduct

            self.A = np.matmul(self.Az, self.G) + self.Au
            self.H = np.matmul(np.matmul(np.transpose(self.G), self.Q), self.G) + self.R
            #test = np.linalg.cholesky(self.H) #To check if H is pos def
        else:
            self.D = np.zeros(self.Nx*(self.Nt+1), self.Nx)
            self.D[0:self.Nx,0:self.Nx] = np.eye(self.Nx)

            self.Ez = np.eye(self.Nx*(self.Nt+1))
            self.Eu = np.zeros(((self.Nt+1)*self.Nx, self.Nt*self.Nu))
            for i in range(self.Nt):
                self.Ez[(i+1)*self.Nx:(i+2)*self.Nx, i*self.Nx:(i+1)*self.Nx] = -Ad
                self.Eu[(i+1)*self.Nx:(i+2)*self.Nx, i*self.Nu:(i+1)*self.Nu] = -Bd
            
            self.A = np.block([[self.Eu, self.Ez],
                               [self.Au, self.Az]])
            self.H = np.block([[self.R, np.zeros((self.R.shape[0],self.Q.shape[1]))],
                               [np.zeros((self.Q.shape[0],self.R.shape[1])), self.Q]])

    def UpdateState(self, currentState):
        # Linearize around x0
        self.UpdateLinearization(currentState)

        # update matrices with new x0 (see licentiate thesis for formulas)
        if self.CONDENSED_MODEL:
            self.f = 2*np.matmul(np.matmul(np.matmul(np.transpose(currentState), np.transpose(self.F)), self.Q), self.G)
            self.bupper = self.b - np.matmul(np.matmul(self.Az, self.F), currentState)
        else:
            self.f = np.full((self.Nt*(self.Nx+self.Nu),), 0)
            self.bupper = np.vstack((np.matmul(self.D, currentState),
                                     self.bIneq))

    def Solve(self, currentState):
        # First update the QP based on the new state
        self.UpdateState(currentState)

        # Solve
        if self.USE_DAQP:
            (uOptimal, fval, exitflag, info) = self.d.quadprog(self.H,self.f,self.A,self.bupper,self.blower,self.sense)
            # TODO: Some exitflag checking
        else:
            if self.FIRST_ITERATION:
                self.FIRST_ITERATION = False
                H = scipy.sparse.csc_matrix(self.H)
                A = scipy.sparse.csc_matrix(self.A)
                self.m.setup(P=H, q=self.f, A=A, l=self.blower, u=self.bupper)
            else:
                self.m.update(q=self.f, u=self.bupper)
            results = self.m.solve()
            uOptimal = results.x
        
        return uOptimal

    def GetControl(self, input):
        '''Returns the first control input'''
        # Transform orientation to roll pitch yaw
        currentState = input.squeeze()
        orientation = Rotation.from_quat(currentState[6:10])
        orientationStates = orientation.as_euler('zyx', degrees=False)
        x0 = np.hstack((currentState[0:3], orientationStates, currentState[3:6], currentState[10:]))

        # Solve
        uOptimal = self.Solve(x0)
        
        return uOptimal[0:6]

    def GetControlSimulation(self,input):
        '''This is only used if the MPC class will be used by the non-nasa simulator'''
        currentState = input.squeeze()
        x0 = currentState

        # Solve
        uOptimal = self.Solve(x0)
        
        return uOptimal[0:6]

    def SetReference(self, xRef):
        # TODO: Make sure that reference is taken into consideration in solver
        pass
