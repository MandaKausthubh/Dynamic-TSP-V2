import numpy as np
import rustworkx as rw
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.primitives import Sampler
from qiskit import Aer
from utils.ClassicalSolver import DynamicGraph

'''using qiskit 0.45 'cause 1.xx is a mess'''
class QuantumQUBOSolver():
    def __init__(self, Q: np.ndarray, c: np.ndarray):
        self.Q = np.array(Q, dtype=float)  # Ensure Q is a float array
        self.c = np.array(c, dtype=float)  # Ensure c is a float array
        self.num_vars = len(c)
        assert len(Q) == self.num_vars
        assert len(Q[0]) == self.num_vars
    
    def _create_hamiltonian(self):
        Pauli_List = []
        n = self.Q.shape[0]
        for i in range(n):
            for j in range(n):  # Include all i,j
                Pauli_Str = ["I"] * n
                Pauli_Str[i], Pauli_Str[j] = "Z", "Z"
                weight = float(self.Q[i,j] / 4)
                Pauli_List.append(("".join(Pauli_Str)[::-1], weight))
        
        for i in range(n):
            Pauli_Str = ["I"] * n
            Pauli_Str[i] = "Z"
            weight = float(-(self.c[i] + np.sum(self.Q[i, :])) / 2)
            Pauli_List.append(("".join(Pauli_Str)[::-1], weight))
        
        return SparsePauliOp.from_list(Pauli_List)

    
    # def solve(self, reps=10):
    #     cost_hamiltonian = self._create_hamiltonian()
        
    #     # Set up optimizer
    #     optimizer = COBYLA(maxiter=1000)
        
    #     # QAOA execution using a sampler
    #     #reps = 20
    #     qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=Sampler())

    #     # Compute minimum eigenvalue
    #     result = qaoa.compute_minimum_eigenvalue(operator=cost_hamiltonian)
        
    #     # Calculate the constant shift
    #     diag_sum = np.sum(np.diag(self.Q))
    #     offdiag_sum = np.sum(self.Q) - diag_sum
    #     shift = diag_sum / 2 + offdiag_sum / 4 + np.sum(self.c) / 2
        
    #     return result, shift

    def solve(self, reps=10):
        cost_hamiltonian = self._create_hamiltonian()
        optimizer = COBYLA(maxiter=1000)
        initial_point = np.random.random(2 * reps)  # Add initial point
        qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=Sampler(), initial_point=initial_point)
        result = qaoa.compute_minimum_eigenvalue(operator=cost_hamiltonian)
        diag_sum = np.sum(np.diag(self.Q))
        offdiag_sum = np.sum(self.Q) - diag_sum
        shift = diag_sum / 2 + offdiag_sum / 4 + np.sum(self.c) / 2
        return result, shift

def results():
    # Dynamic Graph Example
    dynamicGraph = DynamicGraph(3)
    W = np.array([

        [[0,0,0], [1,9,5], [9,5,1]],
        [[1,9,5], [0,0,0], [5,9,1]],
        [[9,5,1], [5,1,9], [0,0,0]],
    ], dtype=np.float64)
    W = np.array(W, dtype=float)
    dynamicGraph.AssignDynamicWeights(W)
    Q, c = dynamicGraph.ConvertToQUBO(1e6)
    result, shift = QuantumQUBOSolver(Q, c).solve()


    # Q = np.ones((3,3))
    # c = np.array([1,4,2])
    # result, shift = QuantumQUBOSolver(Q, c).solve()


    # Print results; add the shift to recover the original QUBO cost.
    print("QAOA Result:")
    print(result)
    # print("Optimal Parameters:", result.optimal_parameters)
    print("Minimum QUBO Cost:", result.eigenvalue + shift)

def test():
    Q = np.array([
        [1, -2],
        [-2, 4]
    ], dtype=np.float64)

    # Define the linear term (bias term)
    c = np.array([-1, 3], dtype=np.float64)
        
    solver = QuantumQUBOSolver(Q, c)
    result, shift = solver.solve()
    
    # Print results; add the shift to recover the original QUBO cost.
    print("QAOA Result:")
    print(result)
    # print("Optimal Parameters:", result.optimal_parameters)
    print("Minimum QUBO Cost:", result.eigenvalue + shift)

# results()
