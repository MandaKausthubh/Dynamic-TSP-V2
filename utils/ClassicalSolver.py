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
        self.num_nodes = num_nodes
        self.DGraph = PyGraph()
        self.DGraph.add_nodes_from(range(num_nodes))

    def AssignDynamicWeights(self, dynamic_weights):
        for indexes, weights in dynamic_weights.items():
            ia, ib = indexes
            self.DGraph.add_edge(ia, ib, weights)

    def convert_to_two(self, ind):
        return (ind // self.num_nodes), (ind % self.num_nodes)

    def GenerateQUBO(self, scalar=1e6):
        Q = np.zeros((self.num_nodes**2, self.num_nodes**2))

        for a in range(self.num_nodes**2):
            for b in range(self.num_nodes**2):
                ia, ta = self.convert_to_two(a)
                ib, tb = self.convert_to_two(b)
                if a == b:
                    Q[a][b] -= 2*scalar
                if tb - ta == 1 and (ia, ib) in self.DGraph.edge_list():
                    Q[a][b] += self.DGraph.get_edge_data(ia, ib)[ta]
                if tb != ta and ia == ib:
                    Q[a][b] += 2*scalar
                if ia != ib and ta == tb:
                    Q[a][b] += 2*scalar
        return Q, 2*scalar*self.num_nodes
            








        