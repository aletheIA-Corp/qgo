from quantum_technology import QuantumTechnology
from genethic_individuals import Individual

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from keras import layers, Model, Input
from qiskit_aer import AerSimulator
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class QGANReproductor:
    def __init__(self, individuals_data: List[Individual], optimizer_executor: QuantumTechnology):
        """
        Initialize the QGAN Hyperparameter Optimizer.
        :param individuals_data (list): List of dictionaries containing hyperparameters and objective function values.
        """
        self.individuals: List[Individual] = individuals_data
        self.optimizer_executor: QuantumTechnology = optimizer_executor
        print(self.individuals)

        self.hyperparameters = [key for key in self.individuals[0].get_individual_values() if key != 'objective_function_values']
        self.num_qubits = len(self.hyperparameters)

        # Extract data matrices
        self.X = np.array([[individual.get_individual_values()[param] for param in self.hyperparameters] for individual in self.individuals])
        self.Y = np.array([[individual.get_individual_values()['objective_function_values']] for individual in self.individuals])

        # Initialize normalization parameters
        self.X_min = self.X.min(axis=0)
        self.X_max = self.X.max(axis=0)
        self.Y_min = self.Y.min()
        self.Y_max = self.Y.max()

        # Normalize data
        self.X_norm = (self.X - self.X_min) / (self.X_max - self.X_min)
        self.Y_norm = (self.Y - self.Y_min) / (self.Y_max - self.Y_min)

        # Setup quantum components
        self.simulator = AerSimulator()
        self.quantum_circuit, self.circuit_parameters = self._create_generator()

        # Setup classical components
        self.discriminator = None

    def _create_generator(self):
        """Create a parameterized quantum circuit for the generator with entanglement."""
        parameters = ParameterVector("θ", self.num_qubits * 2)
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Apply parameterized rotation gates to each qubit
        for i in range(self.num_qubits):
            qc.ry(parameters[i], i)  # Y rotation
            qc.rz(parameters[i + self.num_qubits], i)  # Z rotation

        # Add entanglement between qubits using CNOT gates
        for i in range(self.num_qubits - 1):  # Create entanglement between adjacent qubits
            qc.cx(i, i + 1)  # CNOT between qubit i and qubit i+1

        # Optional: Additional entanglement using a different qubit pair
        if self.num_qubits > 2:
            qc.cx(0, self.num_qubits - 1)  # CNOT between the first and last qubit to create more entanglement

        # Add measurement to each qubit
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        qc.barrier()

        return qc, parameters

    def create_discriminator(self, hidden_layers=None):
        """
        Create the classical discriminator using TensorFlow/Keras.

        Parameters:
        hidden_layers (list): List of integers representing the number of neurons in each hidden layer.
        """
        if hidden_layers is None:
            hidden_layers = [8, 4]  # Default architecture

        inputs = Input(shape=(self.X_norm.shape[1],))
        x = inputs

        for neurons in hidden_layers:
            x = layers.Dense(neurons, activation="relu")(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy"
        )

        self.discriminator = model
        return model

    def train_discriminator(self, batch_size=32, epochs=300, verbose=0):
        """
        Train the discriminator with the normalized data.

        Parameters:
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        verbose (int): Verbosity level (0, 1, or 2).

        Returns:
        History object from the training process.
        """
        if self.discriminator is None:
            self.create_discriminator()

        history = self.discriminator.fit(
            self.X_norm,
            self.Y_norm,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )

        return history

    def evaluate_discriminator(self):
        """
        Evaluate the discriminator on the training data.

        Returns:
        dict: Dictionary containing evaluation metrics.
        """
        if self.discriminator is None:
            raise ValueError("Discriminator has not been created. Call create_discriminator() first.")

        predictions = self.discriminator.predict(self.X_norm)

        mse = np.mean((predictions - self.Y_norm) ** 2)
        mae = np.mean(np.abs(predictions - self.Y_norm))

        return {
            "mse": mse,
            "mae": mae,
            "predictions": predictions,
            "actual": self.Y_norm
        }

    def generate_hyperparameters(self, circuit_parameters_values):
        """
        Execute the quantum circuit with parameters and obtain a vector of normalized hyperparameters.

        Parameters:
        circuit_parameters_values (array): Values for the circuit parameters.

        Returns:
        array: Vector of normalized hyperparameter values.
        """
        # Assign parameter values to the circuit
        parameters_dict = {param: value for param, value in zip(self.circuit_parameters, circuit_parameters_values)}

        # Parameterize the circuit
        parameterized_circuit = self.quantum_circuit.assign_parameters(parameters_dict)

        # Transpile and execute on simulator
        transpiled_circuit = transpile(parameterized_circuit, self.simulator)
        job = self.simulator.run(transpiled_circuit, shots=1024)
        result = job.result()

        # Extract counts from measurement
        counts = result.get_counts()

        # Create array for each hyperparameter
        hyperparameters_array = np.zeros(self.num_qubits)

        # For each qubit/hyperparameter, calculate probability of measuring 1
        for i in range(self.num_qubits):
            prob_one = 0
            total = 0
            for state, count in counts.items():
                # Check if the i-th bit (from right to left) is 1
                if state[-(i + 1)] == '1':  # States are represented as '01', '10', etc.
                    prob_one += count
                total += count

            # Probability of measuring 1 will be our normalized value for this hyperparameter
            hyperparameters_array[i] = prob_one / total if total > 0 else 0.5

        return hyperparameters_array

    def denormalize_hyperparameters(self, normalized_hyperparameters):
        """
        Convert normalized hyperparameter values back to their original scale.

        Parameters:
        normalized_hyperparameters (array): Array of normalized hyperparameter values.

        Returns:
        dict: Dictionary with denormalized hyperparameter values.
        """
        denormalized_values = normalized_hyperparameters * (self.X_max - self.X_min) + self.X_min

        # Convert to integer for hyperparameters that should be integers
        result = {}
        for i, param_name in enumerate(self.hyperparameters):
            # Assuming parameters like n_estimators and max_depth should be integers
            if param_name in ['n_estimators', 'max_depth']:
                result[param_name] = int(round(denormalized_values[i]))
            else:
                result[param_name] = denormalized_values[i]

        return result

    def denormalize_objective(self, normalized_objective):
        """
        Convert normalized objective function value back to its original scale.

        Parameters:
        normalized_objective (float): Normalized objective function value.

        Returns:
        float: Denormalized objective function value.
        """
        return normalized_objective * (self.Y_max - self.Y_min) + self.Y_min

    def generate_and_evaluate_hyperparameters(self, num_samples=50):
        """
        Generate multiple sets of hyperparameters and evaluate them with the discriminator.

        Parameters:
        num_samples (int): Number of hyperparameter sets to generate.

        Returns:
        pandas.DataFrame: DataFrame containing the generated hyperparameters and their predicted objective values.
        """
        if self.discriminator is None:
            raise ValueError("Discriminator has not been created or trained. Call train_discriminator() first.")

        new_hyperparameters = []
        new_predictions = []

        for _ in range(num_samples):
            # Generate random circuit parameters
            test_values = np.random.uniform(-np.pi, np.pi, len(self.circuit_parameters))
            generated_hyperparameters = self.generate_hyperparameters(test_values)

            # Evaluate with discriminator
            generated_prediction = self.discriminator.predict(np.array([generated_hyperparameters]))

            # Store results
            new_hyperparameters.append(generated_hyperparameters)
            new_predictions.append(generated_prediction[0][0])

        # Convert lists to DataFrames
        df_new_hyperparameters = pd.DataFrame(new_hyperparameters, columns=self.hyperparameters)
        df_new_predictions = pd.DataFrame(new_predictions, columns=['Predicted_Objective'])

        # Combine DataFrames
        result_df = pd.concat([df_new_hyperparameters, df_new_predictions], axis=1)

        return result_df

    def get_top_hyperparameters(self, result_df, top_n=10, denormalize=True):
        """
        Get the top n hyperparameter configurations based on predicted objective value.

        Parameters:
        result_df (pandas.DataFrame): DataFrame with hyperparameters and predictions.
        top_n (int): Number of top configurations to return.
        denormalize (bool): Whether to denormalize the hyperparameters.

        Returns:
        pandas.DataFrame: DataFrame with top n configurations.
        """
        # Sort by predicted objective value (assuming higher is better)
        sorted_df = result_df.sort_values(by='Predicted_Objective', ascending=False)
        top_df = sorted_df.head(top_n).copy()

        if denormalize:
            # Create new columns with denormalized values
            for i, param in enumerate(self.hyperparameters):
                denorm_param = f"{param}_denormalized"
                top_df[denorm_param] = top_df[param].apply(
                    lambda x: int(round(x * (self.X_max[i] - self.X_min[i]) + self.X_min[i]))
                    if param in ['n_estimators', 'max_depth']
                    else x * (self.X_max[i] - self.X_min[i]) + self.X_min[i]
                )

            # Denormalize the objective function values
            top_df['Objective_denormalized'] = top_df['Predicted_Objective'].apply(
                lambda x: x * (self.Y_max - self.Y_min) + self.Y_min
            )

        return top_df

    def visualize_results_normalised(self, result_df):
        """
        Visualize the distribution of generated hyperparameters and their predicted objectives.

        Parameters:
        result_df (pandas.DataFrame): DataFrame with hyperparameters and predictions.

        Returns:
        matplotlib.figure.Figure: The generated figure.
        """
        fig = plt.figure(figsize=(12, 10))

        # Plot hyperparameter distributions
        for i, param in enumerate(self.hyperparameters):
            plt.subplot(len(self.hyperparameters) + 1, 1, i + 1)
            plt.hist(result_df[param], bins=20, alpha=0.7)
            plt.title(f"Distribution of {param}")
            plt.xlabel("Normalized Value")
            plt.ylabel("Frequency")

        # Plot objective predictions
        plt.subplot(len(self.hyperparameters) + 1, 1, len(self.hyperparameters) + 1)
        plt.hist(result_df['Predicted_Objective'], bins=20, alpha=0.7)
        plt.title("Distribution of Predicted Objective Values")
        plt.xlabel("Normalized Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        return fig

    def visualize_results_denormalised(self, result_df):
        """
        Visualize the distribution of generated hyperparameters and their predicted objectives (denormalized).

        Parameters:
        result_df (pandas.DataFrame): DataFrame with hyperparameters and predictions.

        Returns:
        matplotlib.figure.Figure: The generated figure.
        """
        # Denormalize hyperparameters
        denormalized_df = result_df.copy()

        for i, param in enumerate(self.hyperparameters):
            denormalized_df[param] = denormalized_df[param].apply(
                lambda x: int(round(x * (self.X_max[i] - self.X_min[i]) + self.X_min[i]))
                if param in ['n_estimators', 'max_depth']
                else x * (self.X_max[i] - self.X_min[i]) + self.X_min[i]
            )

        # Denormalize the objective function values
        denormalized_df['Objective_denormalized'] = denormalized_df['Predicted_Objective'].apply(
            lambda x: x * (self.Y_max - self.Y_min) + self.Y_min
        )

        # Create the figure for plotting
        fig = plt.figure(figsize=(12, 10))

        # Plot denormalized hyperparameter distributions
        for i, param in enumerate(self.hyperparameters):
            plt.subplot(len(self.hyperparameters) + 1, 1, i + 1)
            plt.hist(denormalized_df[param], bins=20, alpha=0.7)
            plt.title(f"Distribution of {param} (Denormalized)")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

        # Plot denormalized objective predictions
        plt.subplot(len(self.hyperparameters) + 1, 1, len(self.hyperparameters) + 1)
        plt.hist(denormalized_df['Objective_denormalized'], bins=20, alpha=0.7)
        plt.title("Distribution of Denormalized Predicted Objective Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        return fig

    def run_optimization_pipeline(self, num_samples=100, top_n=10, discriminator_epochs=300, verbose=0):
        """
        Run the complete optimization pipeline from training to generating top hyperparameters.

        Parameters:
        num_samples (int): Number of hyperparameter sets to generate.
        top_n (int): Number of top configurations to return.
        discriminator_epochs (int): Number of epochs to train the discriminator.
        verbose (int): Verbosity level for training.

        Returns:
        dict: Dictionary containing results and visualizations.
        """
        # Create and train discriminator
        self.create_discriminator()
        history = self.train_discriminator(epochs=discriminator_epochs, verbose=verbose)

        # Evaluate discriminator
        eval_metrics = self.evaluate_discriminator()

        # Generate and evaluate hyperparameters
        result_df = self.generate_and_evaluate_hyperparameters(num_samples=num_samples)

        # Get top hyperparameters
        top_hyperparameters = self.get_top_hyperparameters(result_df, top_n=top_n)

        # Visualize results
        fig1 = self.visualize_results_normalised(result_df)
        fig2 = self.visualize_results_denormalised(result_df)

        return {
            "discriminator_evaluation": eval_metrics,
            "generated_hyperparameters": result_df,
            "top_hyperparameters": top_hyperparameters,
            "visualization_normalised": fig1,
            "visualization_denormalised": fig2,
            "training_history": history
        }


# Example usage
"""if __name__ == "__main__":
    # Sample data
    individuals = [
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

    # Initialize and run the optimizer
    optimizer = QGANReproductor(individuals)
    results = optimizer.run_optimization_pipeline(num_samples=50, top_n=10, discriminator_epochs=300, verbose=1)

    # Display results
    print("\nDiscriminator Evaluation:")
    print(f"MSE: {results['discriminator_evaluation']['mse']}")
    print(f"MAE: {results['discriminator_evaluation']['mae']}")

    print("\nTop 10 Hyperparameter Configurations:")
    print(results['top_hyperparameters'][
              ['n_estimators_denormalized', 'max_depth_denormalized', 'Objective_denormalized']])

    # Show visualizations
    plt.show()"""