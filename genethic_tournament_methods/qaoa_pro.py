"""# Importar librerías necesarias
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

print(num_combinations)
print(combinations)

# 2. Crear el circuito cuántico
nqubits = int(np.ceil(np.log2(num_combinations)))  # Número de qubits necesarios para representar todas las combinaciones
qc = QuantumCircuit(nqubits)
print(nqubits)

# 3. Inicialización: Estado inicial con igual probabilidad para todas las combinaciones
qc.h(range(nqubits))  # Aplicar Hadamard a todos los qubits para generar una superposición uniforme

# 4. Aplicar el operador de coste
# En este caso, el operador de coste penaliza las combinaciones con MAE alto
# Aquí asumimos que conocemos los MAE de algunas combinaciones (esto es un ejemplo simple)
mae_values: dict = {
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

# Obtener los 5 mejores valores de mae_values
sorted_mae_values = sorted(mae_values.items(), key=lambda item: item[1])
best_5_mae_values = dict(sorted_mae_values[:5])

print("Los 5 mejores valores de mae_values son:")
for comb, mae in best_5_mae_values.items():
    print(f"Combinación: {comb}, MAE: {mae}")

# Mapear los MAE a penalizaciones
penalties = {}
min_mae = min(mae_values.values())
for comb, mae in best_5_mae_values.items():
    # Penalizar las combinaciones con MAE más alto
    penalties[comb] = (mae - min_mae)
    print(comb, mae, penalties[comb])

# Crear un diccionario para mapear las combinaciones a índices
combination_indices = {comb: idx for idx, comb in enumerate(best_5_mae_values.keys())}
print(combination_indices)

# Asumimos que el operador de coste es simplemente una fase controlada
for comb, penalty in penalties.items():
    idx = combination_indices[comb]
    qc.rz(penalty, idx)  # Aplicar una rotación de fase (Operador de coste)
    print(f"Combination: {comb}, Index: {idx}, Penalty: {penalty}")

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

# -- Definimos el sampler para ejecutar shots cantidad de veces el circuito cuantico especificado
job = Sampler(AerSimulator()).run([qc], shots=shots)

# -- Lanzamos el job (tarea de ejecución del circuito cuántico) y obtenemos sus resultados
results = job.result()

# -- Accedemos a los valores de las mediciones del circuito cuantico
# Corregido: SamplerV2 devuelve los resultados en un formato diferente
counts = results[0].data.meas.get_counts()

# Obtener los resultados (mediciones) y calcular las probabilidades
probabilities = {key: value / shots for key, value in counts.items()}

# 8. Mostrar los resultados
print("Probabilidades de las combinaciones:")
for state, prob in probabilities.items():
    # Convertir el estado binario de vuelta a las combinaciones (learning_rate, batch_size)
    bin_state = bin(int(state, 2))[2:].zfill(nqubits)
    idx = int(bin_state, 2)  # Convertir el estado binario a índice
    if idx < len(combinations):  # Verificar que el índice está dentro del rango
        lr, bs = combinations[idx]
        print(f"Combinación: {lr}, {bs} - Probabilidad: {prob:.4f}")
    else:
        print(f"Índice {idx} fuera de rango. Estado: {state}")

# Graficar las probabilidades de las combinaciones
valid_states = [state for state in counts if int(state, 2) < len(combinations)]
labels = [f"({combinations[int(state, 2)][0]}, {combinations[int(state, 2)][1]})" for state in valid_states]
prob_values = [probabilities[state] for state in valid_states]

plt.figure(figsize=(12, 6))
plt.bar(labels, prob_values)
plt.xticks(rotation=90)
plt.xlabel('Combinación (Learning Rate, Batch Size)')
plt.ylabel('Probabilidad')
plt.title('Probabilidades de las combinaciones en QAOA')
plt.tight_layout()
plt.show()"""

# Importar librerías necesarias
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

print(f"Total de combinaciones posibles: {num_combinations}")
print("Primeras 5 combinaciones:", combinations[:5])

# 2. Crear el circuito cuántico
nqubits = int(np.ceil(np.log2(num_combinations)))  # Número de qubits necesarios para representar todas las combinaciones
print(f"Número de qubits requeridos: {nqubits}")

# 3. Definir los MAE conocidos para todas las combinaciones (simulamos conocer todos)
# Aquí simulo datos donde combinaciones con learning rate medio (0.01) y batch size medio (64) tienen mejor rendimiento
mae_values = {}
for lr in learning_rates:
    for bs in batch_sizes:
        # Simulamos un patrón donde lr=0.01 y bs=64 tienen mejor rendimiento (valores más bajos)
        # Distancia al "punto óptimo" determina el MAE
        distance = abs(lr - 0.01) + abs(np.log2(bs) - np.log2(64)) / 4
        mae = 0.3 + 0.5 * distance + 0.1 * np.random.rand()  # Base + distancia + ruido
        mae_values[(lr, bs)] = mae

print("\nMAE para cada combinación:")
for i, comb in enumerate(combinations):
    print(f"{i:2d}. {comb}: {mae_values[comb]:.4f}")

# 4. Seleccionar los 10 mejores padres
sorted_mae_values = sorted(mae_values.items(), key=lambda item: item[1])
best_10_parents = dict(sorted_mae_values[:10])

print("\nLos 10 mejores padres son:")
for idx, (comb, mae) in enumerate(best_10_parents.items()):
    print(f"{idx + 1}. Combinación: {comb}, MAE: {mae:.4f}")


# 5. Función para realizar una iteración QAOA mejorada
def apply_qaoa_iteration(qc, parents_dict, iteration, amplification_factor=10.0, max_amplification=100.0):
    """
    Aplica una iteración de QAOA con una mejora en la exploración.

    Args:
        qc: Circuito cuántico
        parents_dict: Diccionario con las combinaciones padres y sus MAE
        amplification_factor: Factor para amplificar las penalizaciones (adaptativo)
    """
    # Aumentar el factor de amplificación para la iteración, asegurando que no se sobrepase el máximo
    amplification_factor = min(amplification_factor, max_amplification)

    # Mapear los MAE a penalizaciones
    penalties = {}
    min_mae = min(parents_dict.values())
    max_mae = max(parents_dict.values())
    range_mae = max_mae - min_mae

    # Calcular penalizaciones normalizadas para todos los padres
    for comb, mae in parents_dict.items():
        # Normalizar y amplificar la penalización
        # normalized_penalty = 1 - (mae - min_mae) / (range_mae + 1e-10)  # Invertimos el rango
        # penalties[comb] = normalized_penalty * amplification_factor * np.pi
        normalized_penalty = (mae - min_mae) / (range_mae + 1e-10)  # Normalizar MAE
        penalties[comb] = normalized_penalty * amplification_factor * np.pi

    # Aplicar operador de coste (RZ) a los padres
    for comb, penalty in penalties.items():
        idx = combinations.index(comb)
        # Convertir el índice a binario
        binary = format(idx, f'0{nqubits}b')

        # Aplicar fase condicional
        for q_idx, bit in enumerate(binary[::-1]):
            if bit == '0':
                qc.x(q_idx)  # Flip si el bit es 0

        # Aplicar fase controlada por todos los qubits
        if nqubits > 1:
            controls = list(range(nqubits - 1))
            qc.crz(penalty, controls, nqubits - 1)
            for i in range(nqubits - 1):
                controls_i = [j for j in range(nqubits) if j != i]
                qc.crz(penalty / (nqubits), controls_i, i)
        else:
            qc.rz(penalty, 0)

        # Deshacer los flips
        for q_idx, bit in enumerate(binary[::-1]):
            if bit == '0':
                qc.x(q_idx)

    # Ajuste dinámico para equilibrar exploración y explotación
    angle = np.pi / 4 + (np.pi / 2) * (iteration / p_iterations)  # Aumentar exploración progresivamente
    for i in range(nqubits):
        qc.rx(angle, i)

    # Añadir entrelazamiento para mejorar la exploración
    for i in range(nqubits - 1):
        qc.cx(i, i + 1)


# 6. Construir el circuito QAOA completo con mejoras
# Crear el circuito
qc = QuantumCircuit(nqubits)

# Inicialización: Estado inicial con igual probabilidad
qc.h(range(nqubits))  # Superposición uniforme

# Aplicar múltiples iteraciones de QAOA
p_iterations = 2  # Aumentamos el número de iteraciones
for iteration in range(p_iterations):
    amplification = 10.0 * (iteration + 1)  # Incrementar amplificación progresivamente
    apply_qaoa_iteration(qc, best_10_parents, iteration, amplification_factor=amplification)

# Medir el circuito cuántico
qc.measure_all()

# 7. Simulación del circuito
simulator = AerSimulator()
shots = 10000  # Aumentar shots para estadísticas más confiables

# Definir el sampler
job = Sampler(simulator).run([qc], shots=shots)
results = job.result()

# Acceder a los valores de las mediciones
counts = results[0].data.meas.get_counts()

# Calcular probabilidades
probabilities = {key: value / shots for key, value in counts.items()}

# 8. Procesar y visualizar resultados
print("\nProbabilidades de las combinaciones más relevantes:")
# Ordenar por probabilidad
sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
selected_combinations = []
selected_maes = []

# Filtrar solo estados válidos (dentro del rango de combinaciones)
valid_results = []
for state, prob in sorted_probs:
    # Convertir el estado binario a índice
    idx = int(state, 2)
    if idx < len(combinations):
        comb = combinations[idx]
        mae = mae_values[comb]
        valid_results.append((state, prob, comb, mae))
        if len(selected_combinations) < 15:  # Tomar solo los 15 más probables para el análisis
            selected_combinations.append(comb)
            selected_maes.append(mae)
            print(f"Combinación: {comb}, MAE: {mae:.4f}, Probabilidad: {prob:.4f}")

# Visualizar comparación entre las combinaciones originales y las seleccionadas por QAOA
plt.figure(figsize=(12, 10))

# Graficar todas las combinaciones
plt.subplot(2, 1, 1)
all_maes = [mae_values[comb] for comb in combinations]
all_labels = [f"({comb[0]}, {comb[1]})" for comb in combinations]
plt.bar(all_labels, all_maes, color='skyblue')
plt.axhline(y=np.mean(all_maes), color='r', linestyle='-', label=f'Media MAE: {np.mean(all_maes):.4f}')
plt.xticks(rotation=90)
plt.xlabel('Todas las Combinaciones (LR, BS)')
plt.ylabel('MAE')
plt.title('MAE de Todas las Combinaciones')
plt.legend()

# Graficar combinaciones seleccionadas
plt.subplot(2, 1, 2)
selected_labels = [f"({comb[0]}, {comb[1]})" for comb in selected_combinations]
plt.bar(selected_labels, selected_maes, color='lightgreen')
plt.axhline(y=np.mean(selected_maes), color='r', linestyle='-', label=f'Media MAE: {np.mean(selected_maes):.4f}')
plt.xticks(rotation=90)
plt.xlabel('Combinaciones Seleccionadas por QAOA (LR, BS)')
plt.ylabel('MAE')
plt.title('MAE de las Combinaciones Seleccionadas por QAOA')
plt.legend()

plt.tight_layout()
plt.show()

# Calcular la mejora
selected_mae_mean = np.mean(selected_maes)
all_mae_mean = np.mean(all_maes)
improvement = ((all_mae_mean - selected_mae_mean) / all_mae_mean) * 100

print("\nAnálisis de eficacia del QAOA:")
print(f"Media MAE de todas las combinaciones: {all_mae_mean:.4f}")
print(f"Media MAE de combinaciones seleccionadas: {selected_mae_mean:.4f}")
print(f"Mejora porcentual: {improvement:.2f}%")

# Filtrar solo estados válidos (dentro del rango de combinaciones)
# Filtrar solo estados válidos (dentro del rango de combinaciones)
valid_results = []
for state, prob in sorted_probs:
    # Convertir el estado binario a índice
    idx = int(state, 2)

    # Filtrar índices fuera del rango de combinaciones disponibles
    if idx < len(combinations):
        comb = combinations[idx]
        mae = mae_values[comb]
        valid_results.append((state, prob, comb, mae))
        if len(selected_combinations) < 15:  # Tomar solo los 15 más probables para el análisis
            selected_combinations.append(comb)
            selected_maes.append(mae)
            print(f"Combinación: {comb}, MAE: {mae:.4f}, Probabilidad: {prob:.4f}")

