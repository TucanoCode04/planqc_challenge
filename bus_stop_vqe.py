
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from rustworkx import layers
from scipy.optimize import minimize


def run_vqe(n, layers, costs, edges, initial_theta):

    terms = []

    for i in range(n):
        label = ["I"] * n
        label[n-1-i] = "Z"
        terms.append(("".join(label), 15 - 2*costs[i]))

    for i, j in edges:
        label = ["I"] * n
        label[n-1-i] = "Z"
        label[n-1-j] = "Z"
        terms.append(("".join(label), 5))

    H = SparsePauliOp.from_list(terms)

    theta = ParameterVector("theta", n * layers)
    qc = QuantumCircuit(n)

    k = 0

    for layer in range(layers):

        for i in range(n):
            qc.ry(theta[k], i)
            k += 1

        for i, j in edges:
            qc.cz(i, j)

    def energy(params):
        bound = qc.assign_parameters(dict(zip(theta, params)))
        state = Statevector.from_instruction(bound)
        return np.real(state.expectation_value(H))

    result = minimize(
        energy,
        initial_theta,
        method="COBYLA",
        options={"maxiter": 30000,
                 "tol": 1e-6}
    )

    optimized_circuit = qc.assign_parameters(
        dict(zip(theta, result.x))
    )

    state = Statevector.from_instruction(optimized_circuit)

    probabilities = state.probabilities()

    most_probable_index = np.argmax(probabilities)

    most_probable_state = format(most_probable_index, f"0{n}b")

    most_probable_probability = probabilities[most_probable_index]

    print("Most probable state:", most_probable_state)
    print("Probability:", most_probable_probability)

    optimal_index = int(most_probable_state, 2)

    optimal_probability = probabilities[optimal_index]

    print("Optimal state probability:", optimal_probability)

    print("Initial theta:", initial_theta)
    print("Optimized theta:", result.x)
    print("Minimum energy:", result.fun)
    print("Minimum cost:", (result.fun + 146) / 4)

    return result, qc, H

n = 12
layers = 1

costs = [2, 1, 3, 4, 2, 1, 3, 2, 4, 1, 3, 2]

edges = [
    (0,1), (1,2), (0,2),
    (3,4), (4,5), (3,5),
    (6,7), (7,8), (6,8),
    (9,10), (10,11), (9,11),
    (0,3), (1,6), (2,9),
    (4,7), (5,10), (8,11)
]

initial_theta = np.random.uniform(-0.1, 0.1, n * layers)
initial_theta[np.abs(initial_theta) < 1e-4] = 1e-4

result, circuit, H = run_vqe(n, layers, costs, edges, initial_theta)
