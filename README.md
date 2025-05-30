# Dynamic TSP QUBO

**DynamicTSPQUBO** is a research project that formulates and solves the Dynamic Travelling Salesman Problem (DTSP) as a Quadratic Unconstrained Binary Optimization (QUBO) model, enabling efficient implementation on quantum annealers and hybrid classical-quantum solvers.

## Table of Contents

* [Background](#background)
* [Problem Formulation](#problem-formulation)
* [Dynamic Extension](#dynamic-extension)
* [Implementation](#implementation)
* [Installation](#installation)
* [Citation](#citation)
* [Contributing](#contributing)


## Background

The Travelling Salesman Problem (TSP) is a classic combinatorial optimization problem. In its dynamic variant (DTSP), edge-cost updates can occur in real time, requiring adaptive routing strategies. QUBO formulations map such problems into binary variables with a quadratic objective, making them amenable to quantum annealing and other specialized solvers.

## Problem Formulation

1. **Decision Variables**: Binary variables $x_{i,t}$ indicate the presence of the agent at node $i$ at time stamp $t$.

2. **Objective**: Minimize total travel cost:

   $$
   \min \sum_{i,j,t} C_{i,j} x_{i,t} x_{j,t+1} + TSP_CONSTRAINT
   $$

3. **Constraints**:

   * Degree constraint: each city has exactly two incident edges.
   * Subtour elimination: ensure a single Hamiltonian cycle.

The above constraints are encoded as penalty terms in the QUBO matrix.

## Dynamic Extension

Our DTSPQUBO extends the static QUBO by allowing:

* **Incremental node insertion**: Recompute local QUBO terms when new nodes arrive.
* **Edge-weight updates**: Update the cost matrix and adjust corresponding QUBO coefficients.

## Implementation

* **Language**: Python 3.8+
* **Dependencies**:

  * `numpy`
  * `qiskit` (for quantum annealing)
  * `networkx`
  * `matplotlib`

```bash
pip install numpy networkx matplotlib qiskit
```

* **Directory Structure**:

  ```
  .
  ├── demo.ipynb               # Example DEMO of usage
  |-- Report.pdf               # Report of the pdf
  ├── utils/                   
  |  |- ClassicalSolver.py     # Problem Encoding for a Dynamic Graph
  |  |- QuantumSolver.py       # Quantum solving of QUBO
  |  |- QuantumSolverV2.py     # Prefer using this as this is faster
  └── README.md                # This file
  ```

## Installation

```bash
git clone https://github.com/MandaKausthubh/DynamicTSPQUBO.git
cd DynamicTSPQUBO
pip install -r requirements.txt
```

## Citation

If you use this work, please cite:

```bibtex
@article{yourname2025dtspsol,
  title   = {Dynamic TSP via QUBO Formulation for Quantum Annealers},
  author  = {Manda Kausthubh and Aaryan Ajith Dev},
  year    = {2025},
  doi     = {10.1234/qcj.2025.001}
}
```

## Contributing

Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests for improvements.

