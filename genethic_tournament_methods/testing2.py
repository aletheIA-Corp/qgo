from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
import matplotlib.pyplot as plt
import numpy as np

# Parámetros de configuración - Ajustar para velocidad
REPS = 2  # Número de repeticiones QAOA
MAX_ITER = 500  # Número máximo de iteraciones
DEBUG = True  # Activa/desactiva mensajes de depuración

def debug_print(msg):
    """Función para imprimir mensajes de depuración solo si DEBUG=True"""
    if DEBUG:
        print(f"[DEBUG] {msg}")

# Definir espacio de búsqueda 5x5 para obtener 25 combinaciones
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]  # Ahora 5 valores
batch_sizes = [16, 32, 64, 128, 256]  # Ahora 5 valores
combinations = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
num_combinations = len(combinations)
debug_print(f"Creadas {num_combinations} combinaciones de hiperparámetros")

# Valores MAE conocidos (ajustados para las nuevas combinaciones)
mae_values = {
    (0.001, 16): 0.6,
    (0.001, 32): 0.5,
    (0.001, 64): 0.55,
    (0.005, 16): 0.45,
    (0.005, 32): 0.4,
    (0.01, 16): 0.4,
    (0.01, 32): 0.35,
    (0.05, 16): 0.3,
    (0.05, 32): 0.25,
    (0.1, 16): 0.2
}

# Normalizar MAE a penalizaciones
min_mae = min(mae_values.values())
penalties = {comb: mae - min_mae for comb, mae in mae_values.items()}

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

# Usar StatevectorSampler (la recomendación V2)
simulator = AerSimulator(method='statevector')
sampler = Sampler()
debug_print("Simulador y sampler configurados")

# Optimizador
optimizer = COBYLA(maxiter=MAX_ITER)
debug_print(f"Iniciando QAOA con {REPS} repeticiones")

# Configuración de QAOA sin punto inicial (QAOA calculará uno adecuado)
qaoa = QAOA(optimizer=optimizer, reps=REPS, sampler=sampler)
debug_print("QAOA configurado correctamente")

# Resolver el problema
debug_print("Resolviendo problema...")
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

try:
    result = qaoa_optimizer.solve(qp)
    print(result.x)
    debug_print("Problema resuelto exitosamente")

    # Procesar resultados
    debug_print("Procesando todas las probabilidades de las combinaciones")
    all_combinations = []

    # Crear un diccionario con las combinaciones y sus probabilidades
    for i, comb in enumerate(combinations):
        probability = result.x[i]  # Extrae la probabilidad de la combinación correspondiente
        all_combinations.append((comb, probability))

    # Mostrar todas las combinaciones y sus probabilidades
    print("\n=== TODAS LAS COMBINACIONES Y PROBABILIDADES ===")
    for comb, prob in all_combinations:
        print(f"Combinación: Learning Rate={comb[0]}, Batch Size={comb[1]}, Probabilidad: {prob}")

    # Visualización
    labels = [f"LR={comb[0]}, BS={comb[1]}" for comb, _ in all_combinations]
    values = [prob for _, prob in all_combinations]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel("Probabilidad en la solución")
    plt.title("Probabilidades de todas las combinaciones en la solución final")
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"Error durante la optimización: {e}")
    # Si fue un error de dimensión del punto inicial, dar información de depuración
    if "dimension" in str(e) and "initial point" in str(e):
        debug_print("Error de dimensión en punto inicial - QAOA requiere 2*REPS parámetros")
        debug_print(f"Con REPS={REPS}, se requieren {2 * REPS} parámetros iniciales")
        print("\nSolución alternativa rápida: usar algoritmo clásico")

        # Resolver el problema de forma clásica como fallback
        best_comb = None
        best_value = float('inf')

        for comb in mae_values:
            if mae_values[comb] < best_value:
                best_value = mae_values[comb]
                best_comb = comb

        print(f"Mejor combinación (método clásico): Learning Rate={best_comb[0]}, Batch Size={best_comb[1]}")
        print(f"MAE: {best_value}")

except Exception as e:
    print(f"Error inesperado: {e}")
