from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np


class QuantumQUBOSolver__V2():
    def __init__(self, Q:np.ndarray, v:np.ndarray = None, c = None):
        self.QUBO = QuadraticProgram()
        self.__dim__ = Q.shape[0]

        if v is None:
            v = np.zeros((self.__dim__))
        if c is None:
            c = 0.0

        # Adding the variables
        for i in range(Q.shape[0]):
            self.QUBO.binary_var(f"x{i}")

        # Extracting Linear and Quadratic Terms
        linear = [v[i] + Q[i, i] for i in range(Q.shape[0])]
        Quadratic_Dict = {
            (f"x{i}", f"x{j}") : Q[i,j] + Q[j,i] for i in range(Q.shape[0]) for j in range(i+1, Q.shape[0])
        }
        self.QUBO.minimize(linear=linear, quadratic=Quadratic_Dict, constant=c)

    def inspectQUBO(self):
        print(self.QUBO.prettyprint())

    def solve(self, reps=15, verbose=True):
        algorithm_globals.random_seed = 10598
        qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0]*(2*reps), reps=reps)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        qaoa_result = qaoa.solve(self.QUBO)
        if verbose:
            print(qaoa_result.prettyprint())

        return qaoa_result
        
















        