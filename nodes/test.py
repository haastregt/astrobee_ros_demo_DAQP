from MPCParams import MPCParams
from MPCSolver import MPCSolver
from simulation import EmbeddedSimEnvironment
import numpy as np

params = MPCParams()
solver = MPCSolver(params)

sim_env = EmbeddedSimEnvironment(model=params.model,
                                 dynamics=params.model.LinearizedDiscreteDynamics,
                                 controller=solver.solve,
                                 time=20)

stateEuler = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sim_env.run(stateEuler)
sim_env.visualize()
solver.PrintComputationalTime()