from rustworkx import PyGraph
import numpy as np

class GraphNode:
    def __init__(self, value):
        self.index = None
        self.value = value

    def __str__(self):
        return f"GraphNode: {self.value} @ index: {self.index}"

class GraphEdge:
    def __init__(self, value):
        self.index = None
        self.value = value

    def __str__(self, t=0):
        return f"{self.value}"

    def __getitem__(self, ind):
        return self.value[ind]


class DynamicGraph:

    def __init__(self, num_nodes):
        """
        Args:
            num_nodes = Number of Nodes in the Graph
        """
        self.num_nodes = num_nodes
        self.Graph = PyGraph()
        self.Graph.add_nodes_from(GraphNode(i) for i in range(num_nodes))

    def AssignDynamicWeights(self, Weights):
        """
        Args:
            WeightFunctions = 2D (Node indexes) list containing the Edge weights as a 
                            function of time.
        """
        print(Weights.shape)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.Graph.add_edge(i, j, GraphEdge(Weights[i, j, :]))

    def convert_to_two_dims(self, x):
        return (x//self.num_nodes), (x % self.num_nodes)

    def convert_to_single(self, i, t):
        return (self.num_nodes * i) + t

    def zero_Qcondition(self, a, b):
        ia, ta = self.convert_to_two_dims(a)
        ib, tb = self.convert_to_two_dims(b)
        return self.Graph.get_edge_data(ia, ib)[ta] if (tb - ta == 1) else 0

    def first_Qcondition(self, a, b):
        ia, ta = self.convert_to_two_dims(a)
        ib, tb = self.convert_to_two_dims(b)
        if (ta != tb):
            return 0
        elif (ia == ib):
            return 1
        else:
            return 2

    def second_Qcondition(self, a, b):
        ia, ta = self.convert_to_two_dims(a)
        ib, tb = self.convert_to_two_dims(b)
        if (ia != ib):
            return 0
        elif (ta == tb):
            return 1
        else:
            return 2

    def ConvertToQUBO(self, Scaling: int):
        Q = np.zeros((self.num_nodes**2, self.num_nodes**2), dtype=np.float32)
        c = np.ones((self.num_nodes)**2)

        for a in range(self.num_nodes**2):
            for b in range(a, self.num_nodes**2):
                # print(f"a: {a} and b: {b}")
                ia, ta = self.convert_to_two_dims(a)
                ib, tb = self.convert_to_two_dims(b)
                S = self.Graph.get_edge_data(ia, ib)[ta] if tb - ta == 1 else 0
                S += 2*Scaling if (ib != ia and ta == tb) else 0
                S += 2*Scaling if (ib == ia and ta != tb) else 0
                Q[a, b] = S

        c = -2*Scaling*c
        return Q, c
                













        


