import numpy as np
import scipy

# Manual implementation
from mpc import MPC

# CVXPYgen implementation
import cvxpy as cp
from cvxpygen import cpg
import pickle

# Acados implementation
from acados_template import AcadosOcp, AcadosOcpSolver

class MPCSolver():
    def __init__(self, params):

        """
        Class for the MPC Solver

        :params.model: dynamical model
        :params.MPC_HORIZON: horizon
        :params.Q: state cost
        :params.P_LQR: final state cost
        :params.R: control cost
        :params.u_lim: control bounds
        :params.x_lim: state bounds
        :params.Xf: terminal set

        :params.method: "manual", "cvxpygen", "acados"           
        :params.solver: "DAQP", "OSQP"
        :params.qp_format: "sparse", "dense"
        """

        self.model = params.model
        self.dt = params.model.dt
        self.Nx, self.Nu = params.model.n, params.model.m
        self.Nt = params.MPC_HORIZON
        self.Q, self.P, self.R = params.Q, params.P_LQR, params.R
        self.ulb, self.uub, self.xlb, self.xub = -params.u_lim, params.u_lim, -params.x_lim, params.x_lim
        self.Xf = params.Xf
        self.method = params.method
        self.solver = params.solver
        self.qp_format = params.qp_format

        self.ref = None

        if self.method == "manual":
            self.problem = MPC(self.model,self.Nt,self.Q,self.P,self.R,self.ulb,self.uub,self.xlb,self.xub,self.Xf)
        elif self.method == "cvxpygen":
            self.problem = self.GenerateSolverCVXPYgen()
        elif self.method == "acados":
            self.problem = self.GenerateSolverAcados()
        else:
            print("The method " + self.method + "is not a viable setting.")

    def solve(self, x0, xref=None):
        x0 = x0.squeeze()
        if self.method == "manual":
            self.problem.SetReference(xref)
            u0 = self.problem.GetControl(x0)
        elif self.method == "cvxpygen":
            self.SolveCVXPYgen(x0, xref)
            u0 = self.problem.var_dict['U'].value[:,0]
        elif self.method == "acados":
            u0 = self.SolveAcados(x0, xref)
        else:
            print("The method " + self.method + "is not a viable setting.")

        return u0

    def GenerateSolverCVXPYgen(self):
        # define variables
        U = cp.Variable((self.Nu, self.Nt), name='U')
        X = cp.Variable((self.Nx, self.Nt+1), name='X')

        # define parameters
        Psqrt = cp.Parameter((self.Nx, self.Nx), name='Psqrt')
        Qsqrt = cp.Parameter((self.Nx, self.Nx), name='Qsqrt')
        Rsqrt = cp.Parameter((self.Nu, self.Nu), name='Rsqrt')
        A = cp.Parameter((self.Nx, self.Nx), name='A')
        B = cp.Parameter((self.Nx, self.Nu), name='B')
        x_init = cp.Parameter(self.Nx, name='x_init')

        # define objective
        objective = cp.Minimize(cp.sum_squares(Psqrt@X[:,self.Nt]) + cp.sum_squares(Qsqrt@X[:,:self.Nt]) + cp.sum_squares(Rsqrt@U))

        # define constraints
        constraints = [X[:,1:] == A@X[:,:self.Nt]+B@U,
                    cp.abs(U) <= 1,
                    X[:,0] == x_init]

        # define problem and gernerate code
        problem = cp.Problem(objective, constraints)
        cpg.generate_code(problem, code_dir='MPC_code', solver=self.solver)

        from MPC_code.cpg_solver import cpg_solve

        with open('MPC_code/problem.pickle', 'rb') as f:
            prob = pickle.load(f)

        prob.register_solve('CPG', cpg_solve)

        prob.param_dict['Psqrt'].value = scipy.linalg.sqrtm(self.P)
        prob.param_dict['Qsqrt'].value = scipy.linalg.sqrtm(self.Q)
        prob.param_dict['Rsqrt'].value = scipy.linalg.sqrtm(self.R)

        return prob

    def SolveCVXPYgen(self,x0,xref):
        self.problem.param_dict['A'].value = self.model.Ad
        self.problem.param_dict['B'].value = self.model.Bd
        self.problem.param_dict['x_init'].value = x0

        self.problem.solve(method='CPG')

    def GenerateSolverAcados(self):
        ocp = AcadosOcp()

        model = self.model.ExportAcadosModel()
        ocp.model = model

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL' #e is for end (timestep N)
        ocp.model.cost_expr_ext_cost = model.x.T @ self.Q @ model.x + model.u.T @ self.R @ model.u
        ocp.model.cost_expr_ext_cost_e = model.x.T @ self.P @ model.x

        #ocp.constraints.C = np.vstack((np.identity(self.Nx),np.zeros((self.Nu,self.Nx))))
        #ocp.constraints.D = np.vstack((np.zeros((self.Nx,self.Nu)),np.identity(self.Nu)))
        #ocp.constraints.ug = np.vstack((self.xub, self.uub))
        #ocp.constraints.lg = np.vstack((self.xlb, self.ulb))

        ocp.constraints.lbu = self.ulb
        ocp.constraints.ubu = self.uub
        ocp.constraints.idxbu = np.arange(self.Nu)

        ocp.constraints.lbx = self.xlb
        ocp.constraints.ubx = self.xub
        ocp.constraints.idxbx = np.arange(self.Nx)

        ocp.constraints.C_e = self.Xf.A
        ocp.constraints.ug_e = self.Xf.b
        ocp.constraints.lg_e = -100000000000000000*np.ones(np.size(self.Xf.b))

        #ocp.constraints.lbx_0 = np.zeros((12,)) # This is just to initialise dimensions
        #ocp.constraints.ubx_0 = np.zeros((12,))
        #ocp.constraints.idxbx_0 = np.arange(self.Nx)
        ocp.constraints.x0 = np.zeros((12,))

        # set options
        ocp.solver_options.qp_solver = self.solver # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE' #'IRK', 'ERK', 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

        # set prediction horizon
        ocp.dims.N = self.Nt
        ocp.solver_options.tf = self.Nt*self.dt # Final time

        ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

        return ocp_solver
        


    def SolveAcados(self, x0, xref):
        return self.problem.solve_for_x0(x0_bar = x0)