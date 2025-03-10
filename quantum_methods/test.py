import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Importaciones de Qiskit
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import Parameter, ParameterVector
from qiskit_algorithms.optimizers import SPSA
from qiskit.visualization import plot_histogram

from genethic_tournament_methods.testing2 import optimizer


class QiskitQRBM:
    def __init__(self, num_visible=4, num_hidden=4):
        """
        Inicializa una Máquina de Boltzmann Cuántica Restringida usando Qiskit

        Args:
            num_visible: Número de qubits visibles (para hiperparámetros y coste)
            num_hidden: Número de qubits ocultos
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_qubits = num_visible + num_hidden

        # Inicializar parámetros de la QRBM
        num_weights = num_visible * num_hidden
        num_visible_bias = num_visible
        num_hidden_bias = num_hidden

        # Crear vectores de parámetros
        self.weights = ParameterVector('w', num_weights)
        self.visible_bias = ParameterVector('v', num_visible_bias)
        self.hidden_bias = ParameterVector('h', num_hidden_bias)

        # Para el backend de simulación
        self.simulator = Aer.get_backend('qasm_simulator')

    def _build_circuit(self, input_data):
        """
        Construye el circuito cuántico para la QRBM

        Args:
            input_data: Datos de entrada normalizados [0,1]

        Returns:
            QuantumCircuit: Circuito de la QRBM
        """
        qc = QuantumCircuit(self.num_qubits, self.num_visible)

        # Inicializar todos los qubits en superposición
        for i in range(self.num_qubits):
            qc.h(i)

        # Codificar datos de entrada en qubits visibles
        for i in range(self.num_visible):
            qc.ry(2 * np.arcsin(input_data[i]), i)

        # Aplicar bias a qubits visibles
        for i in range(self.num_visible):
            qc.ry(self.visible_bias[i], i)

        # Aplicar bias a qubits ocultos
        for i in range(self.num_hidden):
            qc.ry(self.hidden_bias[i], self.num_visible + i)

        # Aplicar pesos entre qubits visibles y ocultos
        weight_idx = 0
        for i in range(self.num_visible):
            for j in range(self.num_hidden):
                h_idx = self.num_visible + j

                # Interacción entre visible y oculto
                qc.cx(i, h_idx)
                qc.rz(self.weights[weight_idx], h_idx)
                qc.cx(i, h_idx)

                weight_idx += 1

        # Mediciones de qubits visibles
        qc.measure(range(self.num_visible), range(self.num_visible))

        return qc

    def get_probabilities(self, params, input_data, shots=1024):
        """
        Obtiene las probabilidades de los qubits visibles

        Args:
            params: Lista de parámetros [weights, visible_bias, hidden_bias]
            input_data: Datos de entrada normalizados
            shots: Número de ejecuciones

        Returns:
            dict: Probabilidades de los estados
        """
        # Desempaquetar parámetros
        weights_flat = params[:self.num_visible * self.num_hidden]
        visible_bias = params[self.num_visible * self.num_hidden:
                              self.num_visible * self.num_hidden + self.num_visible]
        hidden_bias = params[-self.num_hidden:]

        # Construir circuito
        qc = self._build_circuit(input_data)

        # Asignar valores a los parámetros
        param_dict = {}

        # Asignar weights
        for i, w in enumerate(self.weights):
            param_dict[w] = weights_flat[i]

        # Asignar visible bias
        for i, v in enumerate(self.visible_bias):
            param_dict[v] = visible_bias[i]

        # Asignar hidden bias
        for i, h in enumerate(self.hidden_bias):
            param_dict[h] = hidden_bias[i]

        # Enlazar parámetros
        bound_qc = qc.assign_parameters(param_dict)

        # Ejecutar simulación
        job = optimizer.minimize(bound_qc, self.simulator, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Convertir conteos a probabilidades
        probs = {key: value / shots for key, value in counts.items()}

        return probs

    def compute_loss(self, params, training_data):
        """
        Calcula la función de pérdida basada en la energía

        Args:
            params: Lista de parámetros [weights, visible_bias, hidden_bias]
            training_data: Datos de entrenamiento normalizados

        Returns:
            float: Valor de la función de pérdida
        """
        total_energy = 0

        for data_point in training_data:
            probs = self.get_probabilities(params, data_point)

            # Calcular la energía negativa
            # Para cada estado, contribuye a la energía basada en su probabilidad
            energy = 0
            for state, prob in probs.items():
                # Convertir estado binario a vector de bits
                bits = [int(bit) for bit in state]

                # Calcular energía para este estado
                # (Una simplificación para la demostración)
                state_energy = sum(bits) / len(bits)  # Fracción de bits que son 1
                energy -= prob * state_energy

            total_energy += energy

        return total_energy / len(training_data)

    def train(self, training_data, num_iterations=50, learning_rate=0.1, perturbation=0.1):
        # Initializing the optimizer with both learning_rate and perturbation

        """
        Entrena la QRBM

        Args:
            training_data: Datos de entrenamiento normalizados
            num_iterations: Número de iteraciones de entrenamiento
            learning_rate: Tasa de aprendizaje

        Returns:
            list: Parámetros optimizados
        """
        # Inicializar parámetros
        num_params = self.num_visible * self.num_hidden + self.num_visible + self.num_hidden
        initial_params = np.random.uniform(-0.1, 0.1, num_params)

        # Optimizador SPSA (Simultaneous Perturbation Stochastic Approximation)
        optimizer = SPSA(maxiter=num_iterations, learning_rate=learning_rate, perturbation=perturbation)

        # Función objetivo para minimizar
        def objective(params):
            return self.compute_loss(params, training_data)

        # Optimización
        result = optimizer.minimize(objective, initial_params)

        return result.x

    def generate_samples(self, params, base_sample, num_samples=5):
        """
        Genera nuevas muestras basadas en la distribución aprendida

        Args:
            params: Parámetros optimizados
            base_sample: Muestra base (mejor padre)
            num_samples: Número de muestras a generar

        Returns:
            list: Nuevas muestras generadas
        """
        samples = []

        # Obtener distribución de probabilidad
        probs = self.get_probabilities(params, base_sample, shots=1024 * 5)

        # Convertir distribución discreta a muestras
        states = list(probs.keys())
        probabilities = list(probs.values())

        # Generar muestras
        for _ in range(num_samples):
            # Seleccionar un estado basado en su probabilidad
            selected_state = np.random.choice(states, p=probabilities)

            # Convertir el estado a un vector de bits
            sample = [int(bit) for bit in selected_state]

            # Asegurarse de que tenemos num_visible bits
            if len(sample) < self.num_visible:
                sample = [0] * (self.num_visible - len(sample)) + sample

            samples.append(sample)

        return samples


# Funciones auxiliares para generar datos de entrenamiento
def generar_datos_entrenamiento(num_muestras=50):
    """
    Genera datos de entrenamiento sintéticos

    Args:
        num_muestras: Número de muestras a generar

    Returns:
        np.array: Datos de entrenamiento
    """
    datos = []
    for _ in range(num_muestras):
        tasa_aprendizaje = np.random.uniform(0.001, 0.1)
        batch_size = np.random.randint(16, 256)
        epocas = np.random.randint(10, 100)

        # Función simulada para el coste
        # Mejor rendimiento: tasas bajas, batch size moderado, épocas suficientes
        coste = 1.0 / (tasa_aprendizaje * 10) + abs(batch_size - 128) / 128 + 50 / epocas
        coste = np.exp(-coste) * 0.9  # Normalizar a [0, 0.9]

        datos.append([tasa_aprendizaje, batch_size, epocas, coste])
    return np.array(datos)


def normalizar_datos(datos):
    """
    Normaliza los datos al rango [0,1]

    Args:
        datos: Datos a normalizar

    Returns:
        np.array: Datos normalizados
        MinMaxScaler: Objeto para escalar/desescalar
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(datos), scaler


# Ejemplo de uso
if __name__ == "__main__":
    # 1. Generar datos de entrenamiento
    print("Generando datos de entrenamiento...")
    datos_brutos = generar_datos_entrenamiento(100)

    # 2. Normalizar datos
    print("Normalizando datos...")
    datos_norm, scaler = normalizar_datos(datos_brutos)

    # 3. Crear y entrenar la QRBM
    print("Inicializando QRBM...")
    qrbm = QiskitQRBM(num_visible=4, num_hidden=4)

    print("Entrenando QRBM (esto puede tardar varios minutos)...")
    # Para ahorrar tiempo en la demostración, reducimos iteraciones
    params_entrenados = qrbm.train(datos_norm, num_iterations=20, learning_rate=0.05)

    # 4. Encontrar el mejor conjunto de hiperparámetros (padre)
    indice_mejor = np.argmax(datos_brutos[:, 3])  # Índice del mejor coste
    mejor_padre = datos_norm[indice_mejor]
    mejor_padre_original = datos_brutos[indice_mejor]

    print("\nMejor padre encontrado:")
    print(f"Tasa de aprendizaje: {mejor_padre_original[0]:.6f}")
    print(f"Tamaño de batch: {int(mejor_padre_original[1])}")
    print(f"Épocas: {int(mejor_padre_original[2])}")
    print(f"Valor de coste: {mejor_padre_original[3]:.6f}")

    # 5. Generar nuevos hiperparámetros (hijos)
    print("\nGenerando nuevos hiperparámetros...")
    nuevas_muestras_binarias = qrbm.generate_samples(params_entrenados, mejor_padre, num_samples=5)

    # Convertir muestras binarias a valores reales
    nuevos_hijos = []
    for muestra in nuevas_muestras_binarias:
        # Asegurar que la muestra tiene la longitud correcta
        if len(muestra) < 4:
            muestra = [0] * (4 - len(muestra)) + muestra

        # Transformar muestras binarias a valores continuos
        # Esto es una aproximación simple
        valor_continuo = np.array(muestra) / 1.0  # Convertir a float

        # Desescalar para obtener valores originales
        hijo_escalado = scaler.inverse_transform([valor_continuo])[0]
        nuevos_hijos.append(hijo_escalado)

    # 6. Mostrar resultados
    print("\nHijos generados:")
    for i, hijo in enumerate(nuevos_hijos):
        print(f"\nHijo {i + 1}:")
        print(f"Tasa de aprendizaje: {max(0.001, min(0.1, hijo[0])):.6f}")
        print(f"Tamaño de batch: {int(max(16, min(256, hijo[1])))}")
        print(f"Épocas: {int(max(10, min(100, hijo[2])))}")
        print(f"Valor de coste estimado: {max(0, min(1, hijo[3])):.6f}")

    # 7. Visualizar distribución
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(datos_brutos[:, 0], bins=20, alpha=0.5, label='Original')
    plt.hist([h[0] for h in nuevos_hijos], bins=5, alpha=0.7, label='Generados')
    plt.title('Distribución de Tasa de Aprendizaje')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.hist(datos_brutos[:, 1], bins=20, alpha=0.5, label='Original')
    plt.hist([h[1] for h in nuevos_hijos], bins=5, alpha=0.7, label='Generados')
    plt.title('Distribución de Tamaño de Batch')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(datos_brutos[:, 2], bins=20, alpha=0.5, label='Original')
    plt.hist([h[2] for h in nuevos_hijos], bins=5, alpha=0.7, label='Generados')
    plt.title('Distribución de Épocas')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(datos_brutos[:, 3], bins=20, alpha=0.5, label='Original')
    plt.hist([h[3] for h in nuevos_hijos], bins=5, alpha=0.7, label='Generados')
    plt.title('Distribución de Valores de Coste')
    plt.legend()

    plt.tight_layout()
    plt.savefig('qrbm_qiskit_distribucion.png')
    plt.show()

    # 8. Visualizar un circuito de ejemplo
    print("\nMostrando ejemplo del circuito cuántico...")
    qc_example = qrbm._build_circuit(mejor_padre)
    param_dict = {}
    for i, w in enumerate(qrbm.weights):
        param_dict[w] = params_entrenados[i]
    for i, v in enumerate(qrbm.visible_bias):
        param_dict[v] = params_entrenados[qrbm.num_visible * qrbm.num_hidden + i]
    for i, h in enumerate(qrbm.hidden_bias):
        param_dict[h] = params_entrenados[qrbm.num_visible * qrbm.num_hidden + qrbm.num_visible + i]

    bound_qc = qc_example.bind_parameters(param_dict)
    print(bound_qc.draw())