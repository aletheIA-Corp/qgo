from tensorflow.python.keras.utils.version_utils import callbacks
from quantum_technology_executors import QuantumTechnology

from tensorflow.keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from tensorflow.keras import backend as K
from keras import layers, Model, Input
from qiskit_aer import AerSimulator
from typing import List, Dict

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import os

# -- TODO: Hacer que el modelo de discriminador/regresor siempre sea el mismo (hacerle un finetunning)
# -- TODO: Hacer que el discriminador/regresor entrene con todos los individuos de todas las generaciones

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class QGANReproductor:

    def __init__(self, bounds_dict: Dict, individuals_data: List, optimizer_executor: QuantumTechnology, shots: int = 1024, verbose: bool = True):
        """
        Initialize the QGAN Hyperparameter Optimizer.
        :param bounds_dict: (Dict) Lista de bounds que define los límites de las propiedades de cada individuos.
        :param individuals_data: (List) Lista de diccionarios conteniendo propiedades y valores de la función objetivo.
        :param optimizer_executor: (QuantumTechnology) Objeto de QuantumTechnology que ejecuta los circuitos cuánticos.
        :param shots: (int) Objeto de QuantumTechnology que ejecuta los circuitos cuánticos.
        :param verbose: (bool) Verbose para imprimir información extra.
        """

        # -- Instanciamos las variables generales de la clase
        self.bounds_dict: Dict = bounds_dict
        self.individuals: List = individuals_data
        self.optimizer_executor: QuantumTechnology = optimizer_executor
        self.shots: int = shots
        self.verbose: bool = verbose

        # -- Definimos el diccionario de hiperparámetros
        self.hyperparameters: dict = {key:values["type"] for key, values in self.bounds_dict.items() if key != 'objective_function_values' and key != "generation" and key != "malformation"}

        # -- Definimos la cantidad de qubits del circuito cuántico a partir de la cantidad de hiperparámetros
        self.num_qubits = len(self.hyperparameters)

        # -- Extraemos los datos en matrices (propiedades -X- y valores de la función objetivo -Y-)
        self.X = np.array([[individual.get_individual_values()[param] for param in self.hyperparameters.keys()] for individual in self.individuals])
        self.Y = np.array([[individual.get_individual_values()['objective_function_values']] for individual in self.individuals])

        self.individuals_df = pd.DataFrame(self.X, columns=[z for z in self.hyperparameters.keys()])
        self.individuals_df['objective_function_values'] = self.Y

        # -- Inicializamos el proceso de normalización
        self.X_min = self.X.min(axis=0)
        self.X_max = self.X.max(axis=0)
        self.Y_min = self.Y.min()
        self.Y_max = self.Y.max()

        # -- Normalizamos los datos
        self.X_norm = (self.X - self.X_min) / (self.X_max - self.X_min)
        self.Y_norm = (self.Y - self.Y_min) / (self.Y_max - self.Y_min)

        # -- Instaciamos las variables para el pipeline
        self.discriminator = None
        self.quantum_circuit = None
        self.circuit_parameters = None

    def _create_generator(self):
        """Metodo que crea un circuito cuántico parametrizado que actúa como generador de la GAN"""

        # -- Definimos los parámetros a entrenar
        parameters = ParameterVector("θ", self.num_qubits * 2)

        # -- Definimos el circuito cuántico
        qc: QuantumCircuit = QuantumCircuit(self.num_qubits, self.num_qubits)

        for deep_level in range(1, 6):

            # -- Aplicamos rotación a la puertas parametrizada de cada qubit
            for i in range(self.num_qubits):

                # -- Aplicamos rotación en Y
                qc.ry(parameters[i] / deep_level, i)

                # -- Aplicamos rotación en Z
                qc.rz(parameters[i + self.num_qubits] / deep_level, i)

            # -- Añadimos entrelazamiento entre los qubits con una puerta CNOT

            # -- Creamos entrelazamiento entre qubits adyacentes (CNOT between qubit i and qubit i+1)
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)

            # -- En algunos casos cerramos el circulo de entrelazamiento (CNOT between the first and last qubit)
            if self.num_qubits > 2:
                qc.cx(0, self.num_qubits - 1)

        # -- Añadimos medidas a los qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        # -- Añadimos una barrera
        qc.barrier()

        # -- Visualizamos el circuito
        if self.verbose:
            pass
            # qc.draw('mpl')
            # plt.show()

        return qc, parameters

    @staticmethod
    def custom_loss(threshold=0.75, penalty_factor=10.0):

        def loss(y_true, y_pred):
            error = K.abs(y_true - y_pred)
            penalty = penalty_factor * K.square(error)

            # Calcular cuartil 3
            y_pred_sorted = tf.sort(y_pred)
            n = tf.cast(tf.shape(y_pred)[0], tf.float32)
            q3_index = tf.cast(n * threshold, tf.int32)
            q3 = y_pred_sorted[q3_index]

            # Penalización adicional si la predicción está por debajo del cuartil 3
            penalty += K.maximum(0.0, q3 - y_pred) ** 2
            penalty += penalty_factor * K.maximum(0.0, 0.25 - y_pred) ** 3

            # Guardamos la penalización en los logs
            penalty_mean = K.mean(penalty)

            #
            return K.mean(K.binary_crossentropy(y_true, y_pred)) + penalty_mean

        return loss

    def _create_discriminator(self, hidden_layers=None):
        """
        Metodo que crea el discriminador de la GAN utilizando TensorFlow/Keras

        :param hidden_layers: (list) Lista de enteros que representan el numero de neuronas en cada capa oculta.
        """

        # -- Arquitectura por defecto
        if hidden_layers is None:
            hidden_layers = [8, 4]

        # -- Formateamos los datos de entrada
        inputs = Input(shape=(self.X_norm.shape[1],))
        x = inputs

        # -- Agregamos las capas densas
        for neurons in hidden_layers:
            x = layers.Dense(neurons, activation="relu")(x)

        # -- Agregamos la capa de salida
        outputs = layers.Dense(1, activation="sigmoid")(x)

        # -- Creamos y compilamos el modelo
        model = Model(inputs, outputs)

        # Definimos el optimizador
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # Definir ReduceLROnPlateau: reducirá el learning rate si la pérdida no mejora
        self.reduce_lr = ReduceLROnPlateau(
            monitor='loss',  # Monitorea la función de pérdida
            factor=0.5,  # Reduce el learning rate a la mitad
            patience=5,  # Espera 5 épocas sin mejora para reducir el learning rate
            min_lr=1e-6,  # Tasa de aprendizaje mínima
            verbose=1  # Muestra mensajes cuando el learning rate cambie
        )

        # Compilamos el modelo
        model.compile(optimizer=optimizer, loss=self.custom_loss(threshold=0.75))

        # Guardamos el discriminador
        self.discriminator = model

    def _train_discriminator(self, batch_size=32, epochs=300, verbose=0):

        """
        Metodo para entrenar el discriminado de la GAN con los datos normalizados.
        param: batch_size: (int) Batch size (lote en el que se divide y entrenan los datos)
        param: epochs: (int) Numero de épocas (veces que se recorre los datos)
        param: verbose: Nivels de verbose (0, 1, or 2).
        """

        # -- Entrenamos el discriminador
        history = self.discriminator.fit(
            self.X_norm,
            self.Y_norm,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
            # callbacks=[self.reduce_lr, VisualizationCallback(self.discriminator, validation_data=(self.X_norm, self.Y_norm))] if self.verbose else []
        )

        return history

    """def evaluate_discriminator(self) -> Dict:

        # Evaluamos el discriminador sobre los datos de entrenamiento.
        # Return: Diccionario con metricas de evaluación.


        # -- Obtenemos las predicciones
        predictions = self.discriminator.predict(self.X_norm)

        # -- Obtenemos las métricas de mse y mae
        mse = np.mean((predictions - self.Y_norm) ** 2)
        mae = np.mean(np.abs(predictions - self.Y_norm))

        return {
            "mse": mse,
            "mae": mae,
            "predictions": predictions,
            "actual": self.Y_norm
        }"""

    def generate_hyperparameters(self, circuit_parameters_values_list):
        """
        Ejecuta múltiples circuitos cuánticos con diferentes parámetros en una sola sesión y obtiene los valores generados.

        :param circuit_parameters_values_list: (list) Lista de arrays con valores para los parámetros de cada circuito.

        Returns: (array) Matriz con los hiperparámetros generados para cada circuito.
        """

        # -- Crear todos los circuitos con sus respectivos parámetros
        parameterized_circuits = []

        for circuit_parameters_values in circuit_parameters_values_list:
            parameters_dict = {param: value for param, value in zip(self.circuit_parameters, circuit_parameters_values)}
            parameterized_circuit = self.quantum_circuit.assign_parameters(parameters_dict)
            parameterized_circuits.append(parameterized_circuit)

        # -- Ejecutar TODOS los circuitos juntos en la misma sesión
        results = self.optimizer_executor.run(parameterized_circuits, shots=self.shots)

        # -- Extraer resultados y convertirlos a hiperparámetros
        all_hyperparameters = np.zeros((len(circuit_parameters_values_list), self.num_qubits))

        for circuit_idx, counts in enumerate(results):
            for i in range(self.num_qubits):
                prob_one = sum(count for state, count in counts.items() if state[-(i + 1)] == '1')
                total = sum(counts.values())

                # -- Guardar el resultado normalizado para el circuito correspondiente
                all_hyperparameters[circuit_idx, i] = prob_one / total if total > 0 else 0.5

        print("Esperando el resultado...")

        return all_hyperparameters

    def generate_and_evaluate_hyperparameters(self, parameters: list | None = None, num_samples=10):
        """
        Metodo para generar las propiedades de los hijos a partir del generador y evaluarlos por el discriminador.
        :param num_samples: (int) Número de propiedades a generar.

        Returns: (DataFrame) Df que contiene los hiperparámetros generador y las predicciones de los valores objetivo.
        """

        # -- Chequeamos que exista el discriminador
        if self.discriminator is None:
            raise ValueError("El discriminador no se ha creado con éxito (se debe crear con _create_discriminator)")

        # -- Generamos todos los hijos en una sola ejecución del circuito
        generated_hyperparameters = self.generate_hyperparameters(parameters)

        # -- Evaluamos todos los hijos con el discriminador
        generated_predictions = self.discriminator.predict(generated_hyperparameters)
        # mae = np.mean(np.abs(generated_predictions - self.Y_norm))

        # -- Convertimos los resultados en un DataFrame
        df_new_hyperparameters = pd.DataFrame(generated_hyperparameters,
                                              columns=[z for z in self.hyperparameters.keys()])
        df_new_predictions = pd.DataFrame(generated_predictions, columns=['Predicted_Objective'])

        # -- Combinamos los dfs
        result_df = pd.concat([df_new_hyperparameters, df_new_predictions], axis=1)

        # -- Ordenamos el DataFrame por las predicciones y seleccionamos las mejores
        result_df = result_df.sort_values(by='Predicted_Objective', ascending=False).head(num_samples)

        # return result_df, mae
        return result_df

    def get_top_hyperparameters(self, result_df: pd.DataFrame, top_n=10, denormalize=True) -> pd.DataFrame:
        """
        Metodo para obtener las mejores propiedades basado en los valores objetivos predichos

        :param result_df: (DataFrame) Df con las propiedades y predicciones.
        :param top_n: (int) Numero de mejores combinaciones a retornar.
        :param denormalize: (bool) Si se denormalizan las propiedades o no.

        Returns: (pandas.DataFrame) Df con las mejores propiedades.
        """

        # -- TODO: minimize | maximize
        # -- Ordenamos el df por los mejores valores
        sorted_df = result_df.sort_values(by='Predicted_Objective', ascending=False)
        top_df = sorted_df.head(top_n).copy()

        # -- Denormalizamos los datos
        if denormalize:

            # -- Creamos nuevas columnas con valores denormalizados
            for i, param in enumerate([z for z in self.hyperparameters.keys()]):
                denorm_param = f"{param}_denormalized"
                top_df[denorm_param] = top_df[param].apply(
                    lambda x: int(x * (self.X_max[i] - self.X_min[i]) + self.X_min[i])
                    if "int" == self.hyperparameters[param]
                    else float(round(x * (self.X_max[i] - self.X_min[i]) + self.X_min[i], 7))
                )

            # -- Denormalizamos los valores de la función objetivo
            top_df['Objective_denormalized'] = top_df['Predicted_Objective'].apply(
                lambda x: x * (self.Y_max - self.Y_min) + self.Y_min
            )

        return top_df

    def visualize_results_normalised(self, result_df: pd.DataFrame):
        """
        Visualiza la distribución de los hiperparámetros generados y sus objetivos predichos.
        :param result_df: (DataFrame): DataFrame con las propiedades y predicciones.

        Returns:  matplotlib.figure.Figure: La figura generada.
        """

        # -- Creamos la figura
        fig = plt.figure(figsize=(12, 10))

        # -- Ploteamos la distribución de propiedades
        for i, param in enumerate(self.hyperparameters):
            plt.subplot(len(self.hyperparameters) + 1, 1, i + 1)
            plt.hist(result_df[param], bins=20, alpha=0.7)
            plt.title(f"Distribution of {param}")
            plt.xlabel("Normalized Value")
            plt.ylabel("Frequency")

        # -- Ploteamos los valores objetivos
        plt.subplot(len(self.hyperparameters) + 1, 1, len(self.hyperparameters) + 1)
        plt.hist(result_df['Predicted_Objective'], bins=20, alpha=0.7)
        plt.title("Distribution of Predicted Objective Values")
        plt.xlabel("Normalized Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()


    def visualize_results_denormalised(self, result_df: pd.DataFrame):
        """
        Visualiza la distribución de los hiperparámetros generados y sus objetivos predichos (denormalizados)
        :param result_df: (DataFrame): DataFrame con las propiedades y predicciones.

        Returns:  matplotlib.figure.Figure: La figura generada.
        """

        # -- Parámetros denormalizados
        denormalized_df = result_df.copy()

        for i, param in enumerate(self.hyperparameters):
            denormalized_df[param] = denormalized_df[param].apply(
                lambda x: int(round(x * (self.X_max[i] - self.X_min[i]) + self.X_min[i]))
                if param in ['n_estimators', 'max_depth']
                else x * (self.X_max[i] - self.X_min[i]) + self.X_min[i]
            )

        # -- Denormalizamos los valores de la función objetivo
        denormalized_df['Objective_denormalized'] = denormalized_df['Predicted_Objective'].apply(
            lambda x: x * (self.Y_max - self.Y_min) + self.Y_min
        )

        # -- Creamos la figura a plotear
        fig = plt.figure(figsize=(12, 10))

        # -- Ploteamos la distribución de las propiedades denormalizadas
        for i, param in enumerate(self.hyperparameters):
            plt.subplot(len(self.hyperparameters) + 1, 1, i + 1)
            plt.hist(denormalized_df[param], bins=20, alpha=0.7)
            plt.title(f"Distribution of {param} (Denormalized)")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

        # -- Ploteamos las predicción de la función objetivo denormalizada
        plt.subplot(len(self.hyperparameters) + 1, 1, len(self.hyperparameters) + 1)
        plt.hist(denormalized_df['Objective_denormalized'], bins=20, alpha=0.7)
        plt.title("Distribution of Denormalized Predicted Objective Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def run_optimization_pipeline(self, num_samples=100, discriminator_epochs=300, generator_iterations=5, verbose=0):
        """
        Executes the complete optimization pipeline with iterative feedback between generator and discriminator.

        :param num_samples: (int) Number of properties to generate (samples).
        :param discriminator_epochs: (int) Number of epochs for training the discriminator.
        :param generator_iterations: (int) Number of iterations to refine the generator.
        :param verbose: (int) Verbose for console prints.

        Returns: (dict) Dictionary containing results and visualization.
        """
        # Initialize variables to track the optimization process
        top_hyperparameters_df = None
        refined_parameters = None
        best_loss = float('inf')
        result_df: pd.DataFrame | None = None
        augmentation: int = 2

        # 1. Create the initial generator (parametrized quantum circuit)
        self.quantum_circuit, self.circuit_parameters = self._create_generator()

        # 2. Create and initially train the discriminator
        self._create_discriminator()
        initial_history = self._train_discriminator(epochs=discriminator_epochs, verbose=verbose)

        # Main optimization loop
        for iteration in range(generator_iterations):
            if verbose:
                print(f"\n--- Optimization Iteration {iteration + 1} ---")

            # 3. Generate initial parameters
            # If first iteration, use random parameters, else use refined parameters from previous iteration
            parameters = (
                [np.random.uniform(-np.pi, np.pi, len(self.circuit_parameters)) for _ in range(num_samples * 2)]
                if iteration == 0
                else refined_parameters
            )

            # 4. Generate and evaluate offspring properties
            # result_df, mae = self.generate_and_evaluate_hyperparameters(parameters=parameters, num_samples=num_samples)
            result_df = self.generate_and_evaluate_hyperparameters(parameters=parameters, num_samples=num_samples * 2)

            # 5. Select top hyperparameters
            top_hyperparameters_df = self.get_top_hyperparameters(result_df, top_n=num_samples)

            # 6. Get best parameters that generated the top offspring
            best_indices = top_hyperparameters_df.index
            best_parameters = [parameters[idx] for idx in best_indices]

            """# 8. Feedback and refinement mechanism
            if current_loss < best_loss:
                # If current discriminator performance is better, refine parameters
                refined_parameters = [
                    param + np.random.normal(loc=0.0, scale=0.3, size=len(param))  # Adaptive noise scaling
                    for param in best_parameters
                ]
                best_loss = current_loss

                # Optional: Retrain discriminator with new data
                if verbose:
                    print("Retraining discriminator with refined data...")

                # Create a new training set combining original and generated data
                combined_X = np.vstack([self.X_norm, result_df.drop('Predicted_Objective', axis=1).values])
                combined_Y = np.vstack([self.Y_norm, result_df['Predicted_Objective'].values.reshape(-1, 1)])

                # Retrain discriminator with combined data
                self.discriminator.fit(
                    combined_X,
                    combined_Y,
                    epochs=50,  # Shorter retraining
                    verbose=verbose
                )

            # Visualization for verbose mode
            if verbose:
                print("\nTop Hyperparameters:")
                print(top_hyperparameters_df)

                # Generate and show visualizations
                self.visualize_results_normalised(result_df)
                self.visualize_results_denormalised(result_df)
                plt.show()"""

            # If current discriminator performance is better, refine parameters
            refined_parameters = [
                param + np.random.normal(loc=0.0, scale=0.3, size=len(param))  # Adaptive noise scaling
                for param in best_parameters
            ]

            # Optional: Retrain discriminator with new data
            if verbose:
                print("Retraining discriminator with refined data...")

            # Create a new training set combining original and generated data
            combined_X = np.vstack([self.X_norm, result_df.drop('Predicted_Objective', axis=1).values])
            combined_Y = np.vstack([self.Y_norm, result_df['Predicted_Objective'].values.reshape(-1, 1)])

            # Retrain discriminator with combined data
            self.discriminator.fit(
                combined_X,
                combined_Y,
                epochs=50,  # Shorter retraining
                verbose=verbose
                # callbacks=[self.reduce_lr, VisualizationCallback(self.discriminator, validation_data=(self.X_norm, self.Y_norm))] if self.verbose else []
            )

            # Visualization for verbose mode
            if verbose:

                # Generate and show visualizations
                # self.visualize_results_normalised(result_df)
                self.visualize_results_denormalised(result_df)

        # Final results preparation
        # Filtramos solo las columnas que contienen "_denormalized"
        top_hyperparameters = top_hyperparameters_df.filter(like="_denormalized")

        # Renombramos "Objective_denormalized" a "objective_function_values"
        top_hyperparameters = top_hyperparameters.rename(
            columns={"Objective_denormalized": "objective_function_values"}
        )

        # Renombramos también los hiperparámetros para que coincidan con self.individuals_df
        top_hyperparameters = top_hyperparameters.rename(
            columns=lambda x: x.replace("_denormalized", "")
        )

        # Concatenamos los DataFrame
        top_hyperparameters = pd.concat([top_hyperparameters, self.individuals_df], axis=0, ignore_index=True)

        top_hyperparameters = top_hyperparameters.sort_values("objective_function_values", ascending=False)

        print(top_hyperparameters.head(num_samples))
        breakpoint()

        # Eliminamos cualquier columna con "Objective" o "objective"
        top_hyperparameters = top_hyperparameters.loc[:,
                              ~top_hyperparameters.columns.str.contains("Objective|objective", case=False)]

        # Convertimos a diccionario sin la columna "index"
        top_hyperparameters_dict = top_hyperparameters.to_dict()

        # Convert to list of dictionaries
        top_hyperparameters = {
            i: {key: values[i] for key, values in top_hyperparameters_dict.items() if key != "Objective"}
            for i in range(len(next(iter(top_hyperparameters_dict.values()))))
        }

        return {
            "generated_hyperparameters": result_df,
            "top_hyperparameters": top_hyperparameters,
            "training_history": initial_history
        }


class VisualizationCallback(Callback):
    def __init__(self, discriminator, validation_data=None):
        super(VisualizationCallback, self).__init__()
        self.discriminator = discriminator
        self.X_val, self.Y_val = validation_data
        self.loss_history = []
        self.penalty_history = []
        self.fig, self.ax = plt.subplots(1, 2, figsize=(18, 6))

    @staticmethod
    def calculate_penalty(y_true, y_pred, threshold=0.75, penalty_factor=5.0):
        error = np.abs(y_true - y_pred)
        penalty = penalty_factor * np.square(error)

        # -- Calculamos la penalidad con respecto al cuartil 1 y cuartil 3
        q3 = np.percentile(y_pred, threshold * 100)
        penalty += np.maximum(0.0, q3 - y_pred) ** 2
        penalty += penalty_factor * np.maximum(0.0, 0.25 - y_pred) ** 3

        return np.mean(penalty)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # -- Guardamos loss
        if 'loss' in logs:
            self.loss_history.append(logs['loss'])

        # -- Calculamos manualmente la penalización
        y_pred = self.discriminator.predict(self.X_val)
        penalty = self.calculate_penalty(self.Y_val, y_pred)
        self.penalty_history.append(penalty)

        # -- Graficamos la pérdida
        self.ax[0].cla()
        self.ax[0].plot(self.loss_history, label='Train Loss', color='blue', marker='o')
        self.ax[0].set_title(f'Epoch {epoch} - Loss')
        self.ax[0].legend()
        self.ax[0].grid(True)

        # -- Graficamos la penalización
        self.ax[1].cla()
        self.ax[1].plot(self.penalty_history, label='Penalty', color='red', marker='o')
        self.ax[1].set_title(f'Epoch {epoch} - Penalty')
        self.ax[1].legend()
        self.ax[1].grid(True)

        plt.draw()
        plt.pause(0.1)