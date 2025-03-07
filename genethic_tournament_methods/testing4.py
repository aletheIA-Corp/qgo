from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator, Sampler
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Parámetros de configuración
REPS = 2 # Repeticiones para QAOA
MAX_ITER = 50
DEBUG = True

# Definir espacio de búsqueda
learning_rates = [0.001, 0.005, 0.01]
batch_sizes = [16, 32, 64]
combinations = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
num_combinations = len(combinations)

# Valores MAE conocidos (menor es mejor)
mae_values = {
    (0.001, 16): 0.6,
    (0.001, 32): 0.5,
    (0.001, 64): 0.55,
    (0.005, 16): 0.45,
    (0.005, 32): 0.4
}


def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


# Función para normalizar los hiperparámetros
def normalize_hyperparams():
    lr_values = np.array([lr for lr, _ in combinations]).reshape(-1, 1)
    bs_values = np.array([bs for _, bs in combinations]).reshape(-1, 1)

    # Crear escaladores
    lr_scaler = MinMaxScaler()
    bs_scaler = MinMaxScaler()

    # Ajustar escaladores
    normalized_lr = lr_scaler.fit_transform(lr_values)
    normalized_bs = bs_scaler.fit_transform(bs_values)

    # Crear diccionario con valores normalizados
    normalized_params = {}
    for i, (lr, bs) in enumerate(combinations):
        normalized_params[(lr, bs)] = (normalized_lr[i][0], normalized_bs[i][0])

    return normalized_params


# Normalizar hiperparámetros para calcular similaridad
normalized_params = normalize_hyperparams()
debug_print("Parámetros normalizados:")
for comb, norm in normalized_params.items():
    debug_print(f"{comb} -> {norm}")


# Función para calcular similaridad basada en distancia euclidiana
def calculate_similarity(comb1, comb2):
    """
    Calcula la similaridad entre dos combinaciones de hiperparámetros
    0 = completamente diferentes, 1 = idénticos
    """
    norm1 = normalized_params[comb1]
    norm2 = normalized_params[comb2]

    # Distancia euclidiana normalizada
    distance = np.sqrt(((norm1[0] - norm2[0]) ** 2) + ((norm1[1] - norm2[1]) ** 2))

    # Convertir distancia a similaridad (1 - distancia normalizada)
    # Como los parámetros están normalizados, la distancia máxima posible es √2
    similarity = 1 - (distance / np.sqrt(2))

    return similarity


# Ordenar combinaciones conocidas por MAE (mejor a peor)
known_combinations = sorted(mae_values.items(), key=lambda x: x[1])
debug_print("Combinaciones conocidas ordenadas por MAE:")
for comb, mae in known_combinations:
    debug_print(f"{comb}: {mae}")

# Calcular fitness para combinaciones conocidas
max_mae = max(mae_values.values())
min_mae = min(mae_values.values())
mae_range = max_mae - min_mae

# Evitar división por cero si todos los MAE son iguales
if mae_range == 0:
    mae_range = 1

fitness = {comb: (max_mae - mae) / mae_range + 0.1 for comb, mae in mae_values.items()}

# Elegir los N mejores padres para calcular similaridad
N_BEST_PARENTS = 3
best_parents = [comb for comb, _ in known_combinations[:N_BEST_PARENTS]]
debug_print(f"Mejores {N_BEST_PARENTS} padres: {best_parents}")


# Crear hijos con variabilidad alrededor de los mejores padres
def create_children_from_parents(best_parents, num_children=10, variability_percentage=0.2):
    # Obtener límites de los parámetros
    learning_rates = [lr for lr, _ in mae_values.keys()]
    batch_sizes = [bs for _, bs in mae_values.keys()]
    min_lr, max_lr = min(learning_rates), max(learning_rates)
    min_bs, max_bs = min(batch_sizes), max(batch_sizes)

    # Crear hijos alrededor de los mejores padres
    children = []
    for _ in range(num_children):
        # Elegir un hijo cerca del mejor padre (con variabilidad)
        best_lr, best_bs = best_parents[np.random.choice(len(best_parents))]

        # Variabilidad para Learning Rate
        lr_variation = np.random.uniform(-variability_percentage, variability_percentage)
        lr_new = best_lr + lr_variation * best_lr
        lr_new = np.clip(lr_new, min_lr, max_lr)

        # Variabilidad para Batch Size
        bs_variation = np.random.uniform(-variability_percentage, variability_percentage)
        bs_new = best_bs + bs_variation * best_bs
        bs_new = np.clip(bs_new, min_bs, max_bs)

        # Añadir hijo
        children.append((lr_new, bs_new))

    return children


# Generar los hijos
children = create_children_from_parents(best_parents)
debug_print(f"Hijos generados:")
for lr, bs in children:
    debug_print(f"Learning Rate: {lr:.4f}, Batch Size: {bs:.0f}")

# Calcular fitness adaptativo para combinaciones desconocidas
for comb in combinations:
    if comb not in fitness:
        # Calcular similaridad con los mejores padres
        similarities = [calculate_similarity(comb, parent) for parent in best_parents]

        # Pesos de los padres (damos más peso a padres con mejor MAE)
        parent_weights = [(max_mae - mae_values[parent]) / mae_range for parent in best_parents]
        parent_weights = [w / sum(parent_weights) for w in parent_weights]  # Normalizar pesos

        # Calcular similaridad ponderada
        weighted_similarity = sum(sim * weight for sim, weight in zip(similarities, parent_weights))

        # Calcular fitness adaptativo
        adaptive_fitness = 0.2 + 0.3 * weighted_similarity
        fitness[comb] = adaptive_fitness

        debug_print(f"Fitness adaptativo para {comb}: {adaptive_fitness} (similitud: {weighted_similarity})")

# Crear tabla de similaridad para visualización
similarity_table = {}
for comb in combinations:
    if comb not in mae_values:  # Solo para combinaciones desconocidas
        similarity_table[comb] = {
            "fitness": fitness[comb],
            "similarities": {}
        }
        for parent in best_parents:
            similarity_table[comb]["similarities"][parent] = calculate_similarity(comb, parent)

# -- TODO: calcular el fitness para los hijos
# Añadir los hijos a la lista de combinaciones
combinations_with_children = children

# Definir el problema de optimización con todas las combinaciones (padres + hijos)
qp = QuadraticProgram()
for i in range(len(combinations_with_children)):
    qp.binary_var(f"x_{i}")

# IMPORTANTE: Ahora maximizamos en lugar de minimizar
linear_terms = {f"x_{i}": fitness.get(combinations_with_children[i], 0) for i in range(len(combinations_with_children))}
qp.maximize(linear=linear_terms)

# Permitir seleccionar múltiples combinaciones (como en un torneo genético)
qp.linear_constraint(linear={f"x_{i}": 1 for i in range(len(combinations_with_children))},
                     sense="<=", rhs=3)

# Asegurarse de seleccionar al menos una combinación
qp.linear_constraint(linear={f"x_{i}": 1 for i in range(len(combinations_with_children))},
                     sense=">=", rhs=1)

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
    # Para cada muestra, identificar qué índices están activos (=1)
    active_indices = [i for i, val in enumerate(sample.x) if val == 1]
    # Para cada índice activo, sumar su probabilidad a la combinación correspondiente
    for idx in active_indices:
        if idx < len(combinations_with_children):  # Verificación por seguridad
            comb = combinations_with_children[idx]
            if comb in combination_probs:
                combination_probs[comb] += sample.probability / len(active_indices)
            else:
                combination_probs[comb] = sample.probability / len(active_indices)

# Ordenar las combinaciones por probabilidad descendente
sorted_combinations = sorted(combination_probs.items(), key=lambda x: x[1], reverse=True)

# Mostrar los resultados
print("\n=== PROBABILIDADES DE CADA COMBINACIÓN DE HIPERPARÁMETROS ===")
print(f"{'Learning Rate':<15} {'Batch Size':<15} {'Probabilidad':<15} {'MAE':<10} {'Fitness':<10}")
print("-" * 65)

for (lr, bs), prob in sorted_combinations:
    mae = mae_values.get((lr, bs), "Desconocido")
    fit = fitness.get((lr, bs), 0)
    print(f"{lr:<15} {bs:<15} {prob:.6f} {mae:<10} {fit:.4f}")

# Mostrar tabla de similaridad para combinaciones desconocidas
print("\n=== SIMILARIDAD DE COMBINACIONES DESCONOCIDAS CON MEJORES PADRES ===")
print(f"{'Combinación':<20} {'Fitness':<10} " + " ".join([f"{parent}"[:15].ljust(15) for parent in best_parents]))
print("-" * (20 + 10 + 15 * len(best_parents)))

for comb, data in similarity_table.items():
    similarities_str = " ".join([f"{data['similarities'][parent]:.4f}".ljust(15) for parent in best_parents])
    print(f"{str(comb):<20} {data['fitness']:<10.4f} {similarities_str}")

# Visualización: Probabilidad vs Fitness
plt.figure(figsize=(10, 6))
x_vals = [fitness.get(comb, 0) for comb, _ in sorted_combinations]
y_vals = [prob for _, prob in sorted_combinations]
labels = [f"lr={lr}, bs={bs}" for (lr, bs), _ in sorted_combinations]

plt.scatter(x_vals, y_vals, s=100, alpha=0.7)
for i, label in enumerate(labels):
    plt.annotate(label, (x_vals[i], y_vals[i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')

plt.xlabel('Fitness (mayor es mejor)')
plt.ylabel('Probabilidad QAOA')
plt.title('Relación entre Fitness y Probabilidad QAOA')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualización: Similaridad vs Fitness para combinaciones desconocidas
plt.figure(figsize=(12, 6))

# Preparar datos
unknown_combs = [comb for comb in combinations_with_children if comb not in mae_values]
avg_similarities = []
fitnesses = []
labels = []

for comb in unknown_combs:
    avg_sim = sum(similarity_table[comb]["similarities"].values()) / len(best_parents)
    avg_similarities.append(avg_sim)
    fitnesses.append(fitness.get(comb, 0))
    labels.append(f"lr={comb[0]}, bs={comb[1]}")

# Graficar
plt.scatter(avg_similarities, fitnesses, s=100, alpha=0.7)
for i, label in enumerate(labels):
    plt.annotate(label, (avg_similarities[i], fitnesses[i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')

plt.xlabel('Similitud Promedio con Padres')
plt.ylabel('Fitness')
plt.title('Similitud vs Fitness para Combinaciones Desconocidas')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
