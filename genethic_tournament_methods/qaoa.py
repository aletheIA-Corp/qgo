from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np
import matplotlib.pyplot as plt

# 1. Definir el espacio de búsqueda (5 posibilidades para batch size y learning rate)
learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
batch_sizes = [16, 32, 64, 128, 256]

# Convertir las combinaciones posibles en índices
combinations = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
num_combinations = len(combinations)  # Total de combinaciones posibles (5 * 5 = 25)

print("combinations: ", combinations)
print("num_combinations: ", num_combinations)

# 4. Aplicar el operador de coste
# En este caso, el operador de coste penaliza las combinaciones con MAE alto
# Aquí asumimos que conocemos los MAE de algunas combinaciones (esto es un ejemplo simple)
mae_values = {
    (0.001, 16): 0.6,
    (0.001, 32): 0.5,
    (0.001, 64): 0.55,
    (0.005, 16): 0.45,
    (0.005, 32): 0.4,
    (0.005, 64): 0.5,
    (0.01, 16): 0.65,
    (0.01, 32): 0.55,
    (0.01, 64): 0.45,
    (0.02, 16): 0.7,
    (0.02, 32): 0.6,
    (0.02, 64): 0.55,
    (0.05, 16): 0.75,
    (0.05, 32): 0.65,
    (0.05, 64): 0.6,
}

# 2. Crear el circuito cuántico
# nqubits = int(np.log2(num_combinations)) + 1  # Número de qubits necesarios para representar todas las combinaciones
nqubits = int(len([z for z in mae_values.keys()]))  # Número de qubits necesarios para representar todas las combinaciones

print("nqubits: ", nqubits)

qc = QuantumCircuit(nqubits)

# 3. Inicialización: Estado inicial con igual probabilidad para todas las combinaciones
qc.h(range(nqubits))  # Aplicar Hadamard a todos los qubits para generar una superposición uniforme



# Mapear los MAE a penalizaciones
penalties = {}
min_mae = min(mae_values.values())
for comb, mae in mae_values.items():
    # Penalizar las combinaciones con MAE más alto
    penalties[comb] = (mae - min_mae)
    print("comb, mae, penalties: ", comb, mae, penalties[comb])

# Asumimos que el operador de coste es simplemente una fase controlada
for idx, (lr, bs) in enumerate([z for z in mae_values.keys()]):
    penalty = penalties.get((lr, bs), 0)
    qc.rz(penalty, idx)  # Aplicar una rotación de fase (Operador de coste)

# 5. Aplicar el operador de mezcla
# Esto busca explorar combinaciones cercanas al aplicar puertas X o Y a los qubits
for i in range(nqubits):
    qc.rx(np.pi / 2, i)  # Aplicar una rotación X (como una mezcla)

# 6. Medir el circuito cuántico
qc.measure_all()

# 7. Simulador cuántico
simulator = AerSimulator()

# Ejecutar el circuito
shots = 1024  # Número de mediciones (experimentos cuánticos)

simlator = Sampler(AerSimulator())

result = simlator.run([qc], shots=shots).result()

# Obtener los resultados (mediciones) y calcular las probabilidades
counts = result[0].data.meas.get_counts()
probabilities = {key: value / shots for key, value in counts.items()}

print("counts: ", counts)
print("probabilities: ", probabilities)

# 8. Mostrar los resultados
print("Probabilidades de las combinaciones:")
for state, prob in probabilities.items():
    # Convertir el estado binario de vuelta a las combinaciones (learning_rate, batch_size)
    bin_state = bin(int(state, 2))[2:].zfill(nqubits)
    print(bin_state)
    idx = int(bin_state, 2)  # Convertir el estado binario a índice
    print(idx)
    lr, bs = combinations[idx]
    print(f"Combinación: {lr}, {bs} - Probabilidad: {prob:.4f}")

# Graficar las probabilidades de las combinaciones
labels = [f"({combinations[int(bin(state, 2), 2)][0]}, {combinations[int(bin(state, 2), 2)][1]})" for state in counts]
prob_values = list(probabilities.values())

plt.bar(labels, prob_values)
plt.xticks(rotation=90)
plt.xlabel('Combinación (Learning Rate, Batch Size)')
plt.ylabel('Probabilidad')
plt.title('Probabilidades de las combinaciones en QAOA')
plt.show()