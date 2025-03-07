from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator, Sampler
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de configuración
REPS = 2  # Aumentado para mejor convergencia
MAX_ITER = 100
DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# Definir espacio de búsqueda
learning_rates = [0.001, 0.005, 0.01]
batch_sizes = [16, 32, 64]
combinations = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
num_combinations = len(combinations)
debug_print(f"Creadas {num_combinations} combinaciones de hiperparámetros")

# Valores MAE conocidos
mae_values = {
    (0.001, 16): 0.6,
    (0.001, 32): 0.5,
    (0.001, 64): 0.55,
    (0.005, 16): 0.45,
    (0.005, 32): 0.4
}

# Normalizar MAE a penalizaciones
min_mae = min(mae_values.values())
penalties = {comb: mae - min_mae for comb, mae in mae_values.items()}

# Asignar penalización más baja y agregar un "bonus" para incentivar la exploración de nuevas combinaciones
for comb in combinations:
    if comb not in penalties:
        penalties[comb] = 0.1  # Penalización baja para combinaciones desconocidas
        min_known_mae = min(mae_values.values(), default=1.0)

        # Si el MAE de una combinación desconocida es potencialmente mejor, darle un pequeño bono
        improvement_potential = min_known_mae - 0.4  # Ajustar este valor según el contexto
        penalties[comb] += improvement_potential

# Definir problema de optimización
qp = QuadraticProgram()
for i in range(num_combinations):
    qp.binary_var(f"x_{i}")

# Función objetivo
linear_terms = {f"x_{i}": penalties.get(combinations[i], 1.0) for i in range(num_combinations)}
qp.minimize(linear=linear_terms)

# Restricción para elegir exactamente una combinación
qp.linear_constraint(linear={f"x_{i}": 1 for i in range(num_combinations)}, sense="==", rhs=1)
debug_print("Problema cuadrático configurado")

# Convertir a QUBO
converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)

# Configurar QAOA
sampler = Sampler()
debug_print("Simulador y sampler configurados")

# Optimizador
optimizer = COBYLA(maxiter=MAX_ITER)
debug_print(f"Iniciando QAOA con {REPS} repeticiones")

# Configuración de QAOA
qaoa = QAOA(optimizer=optimizer, reps=REPS, sampler=sampler)

# Resolver con QAOA
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
debug_print("Resolviendo problema con QAOA...")
result = qaoa_optimizer.solve(qubo)
debug_print("Problema resuelto exitosamente")

# Extraer las probabilidades de cada combinación individual
combination_probs = {}

# Iterar por todas las muestras obtenidas
for sample in result.samples:
    active_indices = [i for i, val in enumerate(sample.x) if val == 1]
    for idx in active_indices:
        if idx < len(combinations):  # Verificación por seguridad
            comb = combinations[idx]
            if comb in combination_probs:
                combination_probs[comb] += sample.probability / len(active_indices)
            else:
                combination_probs[comb] = sample.probability / len(active_indices)

# Ordenar las combinaciones por probabilidad descendente
sorted_combinations = sorted(combination_probs.items(), key=lambda x: x[1], reverse=True)

# Mostrar los resultados
print("\n=== PROBABILIDADES DE CADA COMBINACIÓN DE HIPERPARÁMETROS ===")
print(f"{'Learning Rate':<15} {'Batch Size':<15} {'Probabilidad':<15} {'MAE':<10}")
print("-" * 55)

for (lr, bs), prob in sorted_combinations:
    mae = mae_values.get((lr, bs), "Desconocido")
    print(f"{lr:<15} {bs:<15} {prob:.6f} {mae}")

# Visualización
plt.figure(figsize=(12, 6))
labels = [f"lr={c[0]}, bs={c[1]}" for c, _ in sorted_combinations]
values = [prob for _, prob in sorted_combinations]

# Crear gráfico de barras horizontal para mejor visualización
y_pos = np.arange(len(labels))
plt.barh(y_pos, values)
plt.yticks(y_pos, labels)
plt.xlabel('Probabilidad')
plt.title('Probabilidad de cada combinación de hiperparámetros')
plt.tight_layout()
plt.show()

# También mostrar el MAE esperado para las combinaciones conocidas
plt.figure(figsize=(10, 5))
known_combinations = [(c, p) for c, p in sorted_combinations if c in mae_values]
labels = [f"lr={c[0]}, bs={c[1]}" for c, _ in known_combinations]
mae_vals = [mae_values[c] for c, _ in known_combinations]

y_pos = np.arange(len(labels))
plt.barh(y_pos, mae_vals)
plt.yticks(y_pos, labels)
plt.xlabel('MAE (menor es mejor)')
plt.title('MAE de las combinaciones conocidas')
plt.tight_layout()
plt.show()
