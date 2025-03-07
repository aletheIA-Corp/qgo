from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt

# Definir el espacio de búsqueda
learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
batch_sizes = [16, 32, 64, 128, 256]
combinations = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
num_combinations = len(combinations)  # 25 combinaciones posibles

# Definir valores MAE conocidos
mae_values = {
    (0.001, 16): 0.6,
    (0.001, 32): 0.5,
    (0.001, 64): 0.55,
    (0.005, 16): 0.45,
    (0.005, 32): 0.4
    # (0.005, 64): 0.5,
    # (0.01, 16): 0.65,
    # (0.01, 32): 0.55,
    # (0.01, 64): 0.45,
    # (0.02, 16): 0.7,
    # (0.02, 32): 0.6,
    # (0.02, 64): 0.55,
    # (0.05, 16): 0.75,
    # (0.05, 32): 0.65,
    # (0.05, 64): 0.6,
}

# **Corregimos el número de qubits**
nqubits = int(np.ceil(np.log2(num_combinations)))  # log2(25) ≈ 5

# Crear el circuito cuántico
qc = QuantumCircuit(nqubits)
qc.h(range(nqubits))  # Superposición uniforme

# Aplicar operador de coste
penalties = {}
min_mae = min(mae_values.values())
for comb, mae in mae_values.items():
    penalties[comb] = mae - min_mae

for comb, penalty in penalties.items():
    if comb in combinations:
        idx = combinations.index(comb)  # Obtener índice correcto dentro de 25 combinaciones
        bin_idx = format(idx, f'0{nqubits}b')  # Convertir a binario con padding

        for i, bit in enumerate(bin_idx):
            if bit == "1":
                qc.rz(penalty, i)  # Aplicamos penalización a los qubits activos

# Aplicar operador de mezcla
# Operador de mezcla mejorado con varias iteraciones
num_layers = 3  # Número de iteraciones de mezcla y coste

for _ in range(num_layers):
    # Aplicar operador de coste (penalización)
    for comb, penalty in penalties.items():
        if comb in combinations:
            idx = combinations.index(comb)
            bin_idx = format(idx, f'0{nqubits}b')

            for i, bit in enumerate(bin_idx):
                if bit == "1":
                    qc.rz(penalty / num_layers, i)  # Dividir penalización en capas

    # Aplicar operador de mezcla (exploración)
    for i in range(nqubits):
        qc.ry(np.pi / 4, i)  # Rotación en Y para mejor exploración

# Medir
qc.measure_all()

# Simulador cuántico
simulator = AerSimulator()
shots = 1024  # Número de mediciones

# Ejecutar el circuito
result = simulator.run(qc, shots=shots).result()
counts = result.get_counts()

# Convertir mediciones a combinaciones (learning_rate, batch_size)
collapsed_results = {}
for bitstring, count in counts.items():
    index = int(bitstring, 2) % num_combinations
    lr, bs = combinations[index]
    collapsed_results[(lr, bs)] = collapsed_results.get((lr, bs), 0) + count

# Normalizar a probabilidades
total_counts = sum(collapsed_results.values())
probabilities = {k: v / total_counts for k, v in collapsed_results.items()}

# Graficar distribución de probabilidad
plt.figure(figsize=(10, 5))
plt.bar(range(len(probabilities)), probabilities.values(), tick_label=[str(k) for k in probabilities.keys()])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Learning Rate - Batch Size")
plt.ylabel("Probabilidad")
plt.title("Distribución de Probabilidad de Configuraciones")
plt.show()
