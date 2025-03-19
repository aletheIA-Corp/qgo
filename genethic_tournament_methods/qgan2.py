import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import numpy as np
import tensorflow as tf
import qiskit
from qiskit_aer import AerSimulator, Aer
from qiskit import transpile
import os
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import layers

# Datos iniciales
individuos = [
    {'n_estimators': 150, 'max_depth': 2, 'objective_function_values': 0.7528089887640449},
    {'n_estimators': 100, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
    {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 150, 'max_depth': 2, 'objective_function_values': 0.7528089887640449},
    {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
    {'n_estimators': 100, 'max_depth': 4, 'objective_function_values': 0.7191011235955056}
]

# Extraemos los nombres de los hiperparámetros (excluyendo la función objetivo)
hiperparametros = [key for key in individuos[0] if key != 'objective_function_values']

# Convertimos los datos en matrices NumPy
X = np.array([[individuo[param] for param in hiperparametros] for individuo in individuos])
Y = np.array([[individuo['objective_function_values']] for individuo in individuos])

print("Hiperparámetros considerados:", hiperparametros)
print("Shape de X:", X.shape)  # Esperado: (11, 2) -> 11 individuos, 2 hiperparámetros
print("Shape de Y:", Y.shape)  # Esperado: (11, 1) -> 11 pérdidas
print("Ejemplo de X:", X[:3])  # Primeras 3 filas
print("Ejemplo de Y:", Y[:3])  # Primeras 3 pérdidas

# Normalización Min-Max
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)  # Normalizamos X

Y_min = Y.min()
Y_max = Y.max()
Y_norm = (Y - Y_min) / (Y_max - Y_min)  # Normalizamos Y

print("Shape de X_norm:", X_norm.shape)  # Esperado: (11, 2)
print("Shape de Y_norm:", Y_norm.shape)  # Esperado: (11, 1)
print("Ejemplo de X_norm:", X_norm[:3])  # Primeras 3 filas normalizadas
print("Ejemplo de Y_norm:", Y_norm[:3])  # Primeras 3 pérdidas normalizadas


def crear_generador(num_qubits):
    """Crea un generador cuántico basado en un circuito parametrizado."""
    parametros = ParameterVector("θ", num_qubits * 2)  # Parámetros libres
    qc = QuantumCircuit(num_qubits, num_qubits)  # Añadir qubits de medición

    # Aplicamos puertas de rotación parametrizadas en cada qubit
    for i in range(num_qubits):
        qc.ry(parametros[i], i)  # Rotación en Y
        qc.rz(parametros[i + num_qubits], i)  # Rotación en Z

    # Agregamos la medición a cada qubit
    qc.measure(range(num_qubits), range(num_qubits))

    qc.barrier()

    return qc, parametros


# Definimos el número de qubits igual al número de hiperparámetros
num_qubits = X_norm.shape[1]
circuito_generador, parametros_generador = crear_generador(num_qubits)

print(circuito_generador.parameters)

# Mostramos el circuito
print(circuito_generador)


def crear_discriminador(num_inputs):
    """Crea el discriminador clásico usando TensorFlow/Keras con una mejor estructura."""
    inputs = keras.Input(shape=(num_inputs,))
    x = layers.Dense(8, activation="relu")(inputs)
    x = layers.Dense(4, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    modelo = keras.Model(inputs, outputs)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy"
    )

    return modelo


# Creamos el discriminador con la nueva arquitectura
discriminador = crear_discriminador(X_norm.shape[1])

# Mostramos el resumen del modelo
discriminador.summary()

# Simulador cuántico
simulador = AerSimulator()


def generar_hiperparametros(circuito: QuantumCircuit, valores_parametros):
    """Ejecuta el circuito cuántico con parámetros y obtiene un vector de hiperparámetros normalizados."""
    # Asignamos los valores a los parámetros del circuito
    parametros_dict = {parametro: valor for parametro, valor in zip(circuito.parameters, valores_parametros)}

    # Asignamos los valores a los parámetros correctamente
    circuito_parametrizado = circuito.assign_parameters(parametros_dict)

    # Transpilamos y ejecutamos el circuito en el simulador
    circuito_transpilado = transpile(circuito_parametrizado, simulador)
    job = simulador.run(circuito_transpilado, shots=1024)
    resultado = job.result()

    # Extraemos las cuentas de los estados medidos
    counts = resultado.get_counts()

    # Creamos un array para almacenar los valores generados para cada hiperparámetro
    hiperparametros_array = np.zeros(num_qubits)

    # Para cada qubit/hiperparámetro, calculamos la probabilidad de medir 1
    for i in range(num_qubits):
        prob_uno = 0
        total = 0
        for estado, count in counts.items():
            # Verificamos si el i-ésimo bit (de derecha a izquierda) es 1
            if estado[-(i + 1)] == '1':  # Los estados se representan como '01', '10', etc.
                prob_uno += count
            total += count

        # La probabilidad de medir 1 será nuestro valor normalizado para este hiperparámetro
        hiperparametros_array[i] = prob_uno / total if total > 0 else 0.5

    return hiperparametros_array


# Probamos el generador con parámetros aleatorios
valores_prueba = np.random.uniform(-np.pi, np.pi, len(parametros_generador))
hiperparametros_generados = generar_hiperparametros(circuito_generador, valores_prueba)

print("Hiperparámetros generados (normalizados):", hiperparametros_generados)

# Entrenamiento del discriminador
batch_size = 32
epochs = 300

# Entrenamos el discriminador con los datos normalizados
discriminador.fit(X_norm, Y_norm, batch_size=batch_size, epochs=epochs)

# Probamos el discriminador con los mismos datos de entrenamiento
predicciones = discriminador.predict(X_norm)

# Mostramos las predicciones del discriminador
print("Valores reales de la función objetivo:", Y_norm)
print("Predicciones del discriminador:", predicciones)

mse = np.mean((predicciones - Y_norm) ** 2)
mae = np.mean(np.abs(predicciones - Y_norm))
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Error absoluto medio (MAE): {mae}")

# Generar nuevos hiperparámetros con el generador cuántico
nuevos_valores = np.random.uniform(-np.pi, np.pi, len(parametros_generador))
hiperparametros_generados = generar_hiperparametros(circuito_generador, nuevos_valores)

# Evaluar los hiperparámetros generados con el discriminador
predicciones_generadas = discriminador.predict(np.array([hiperparametros_generados]))

# Mostrar los resultados
print("Hiperparámetros generados:", hiperparametros_generados)
print("Predicciones del discriminador sobre los nuevos hiperparámetros:", predicciones_generadas)

# Generar 1000 nuevos conjuntos de hiperparámetros
nuevos_hiperparametros = []
predicciones_nuevas = []

for _ in range(50):
    valores_prueba = np.random.uniform(-np.pi, np.pi, len(parametros_generador))
    hiperparametros_generados = generar_hiperparametros(circuito_generador, valores_prueba)

    # Evaluamos con el discriminador
    prediccion_generada = discriminador.predict(np.array([hiperparametros_generados]))

    # Almacenamos los resultados
    nuevos_hiperparametros.append(hiperparametros_generados)
    predicciones_nuevas.append(prediccion_generada[0][0])

# Convertimos las listas a un DataFrame
df_nuevos_hiperparametros = pd.DataFrame(nuevos_hiperparametros, columns=hiperparametros)
df_nuevas_predicciones = pd.DataFrame(predicciones_nuevas, columns=['Predicción Función Objetivo'])

# Unimos los DataFrames
df_resultado = pd.concat([df_nuevos_hiperparametros, df_nuevas_predicciones], axis=1)

# Seleccionamos los 10 mejores hiperparámetros basados en la predicción de la función objetivo
df_resultado = df_resultado.sort_values(by='Predicción Función Objetivo', ascending=False)
df_mejores = df_resultado.head(10)

# Visualización de la distribución de los hiperparámetros generados
plt.figure(figsize=(12, 6))

# Gráfico de las distribuciones de los hiperparámetros
plt.subplot(1, 2, 1)
plt.hist(df_nuevos_hiperparametros.values, bins=20, alpha=0.7, label="Nuevos hiperparámetros")
plt.title("Distribución de los Hiperparámetros Generados")
plt.xlabel("Valor de los Hiperparámetros")
plt.ylabel("Frecuencia")
plt.legend()

# Gráfico de las predicciones de la función objetivo
plt.subplot(1, 2, 2)
plt.hist(predicciones_nuevas, bins=20, alpha=0.7, label="Predicciones de la función objetivo")
plt.title("Distribución de las Predicciones de la Función Objetivo")
plt.xlabel("Valor de la Función Objetivo")
plt.ylabel("Frecuencia")
plt.legend()

plt.tight_layout()
plt.show()

# Mostrar los 10 mejores hiperparámetros generados
print("Top 10 mejores hiperparámetros generados:")
print(df_mejores)
