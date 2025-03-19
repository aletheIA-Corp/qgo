from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import tensorflow as tf

import keras
import numpy as np


class QuantumLayer(keras.layers.Layer):
    def __init__(self, num_qubits, shots=1000):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = AerSimulator()

        # Trainable parameters in TensorFlow
        self.theta = self.add_weight(name="theta",
                                     shape=(self.num_qubits,),
                                     initializer="random_normal",
                                     trainable=True)

    def build_quantum_circuit(self, thetas):
        """Creates a quantum circuit with parameterized rotations."""
        print("Building quantum circuit with thetas:", thetas)  # Debugging line
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initial superposition
        qc.h(range(self.num_qubits))

        # Apply RX rotations with trainable parameters
        for qubit in range(self.num_qubits):
            qc.rx(thetas[qubit], qubit)

        # Measure qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def quantum_simulation(self, inputs):
        """Runs quantum simulation and gets expectation value."""
        batch_size = inputs.shape[0]
        outputs = []

        for i in range(batch_size):
            # Use a portion of the input to influence the circuit parameters
            # This ensures the quantum circuit is influenced by the input
            input_sample = inputs[i]
            thetas = self.theta.numpy() * (1.0 + 0.1 * tf.reduce_mean(input_sample).numpy())
            print(f"Input sample {i}: {input_sample}, Adjusted thetas: {thetas}")  # Debugging line

            circuit = self.build_quantum_circuit(thetas)

            # Run in the simulator
            transpiled_circuit = transpile(circuit, self.backend)
            job = self.backend.run(transpiled_circuit, shots=self.shots)
            result = job.result().get_counts()

            # Calculate expectation value
            expectation_value = sum([(-1) ** sum(map(int, key)) * val for key, val in result.items()]) / self.shots
            outputs.append([expectation_value])

        return np.array(outputs, dtype=np.float32)

    def call(self, inputs):
        """Calls quantum simulation within TensorFlow."""
        outputs = tf.py_function(self.quantum_simulation, [inputs], tf.float32)

        # Ensure tensor has the correct shape for the following layers
        outputs.set_shape((None, 3))  # Match the output shape expected by the following layers
        return outputs


def build_generator(latent_dim, output_dim=3):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(latent_dim,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(output_dim, activation="tanh")  # Ensures output is in range [-1, 1]
    ])
    return model


def build_discriminator(input_dim):
    """Quantum discriminator that adapts to the input dimension"""
    inputs = keras.Input(shape=(input_dim,))

    # Apply preprocessing to reduce dimensions if needed
    processed = keras.layers.Dense(3, activation="relu")(inputs)

    # Debug: Verifica las dimensiones antes de aplicar el QuantumLayer
    print(f"Shape before QuantumLayer: {processed.shape}")  # Agrega este print

    # Apply quantum layer
    quantum_output = QuantumLayer(num_qubits=3)(processed)

    # Debug: Verifica las dimensiones después de la capa cuántica
    print(f"Shape after QuantumLayer: {quantum_output.shape}")  # Agrega este print

    # Final classification
    dense_output = keras.layers.Dense(1, activation="sigmoid")(quantum_output)

    model = keras.Model(inputs=inputs, outputs=dense_output)
    return model


class QGAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(QGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        # Use from_logits=False since we have sigmoid activation
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)

    def compile(self, gen_optimizer, disc_optimizer):
        super(QGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        # Generate fake data
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_data = self.generator(random_latent_vectors, training=True)

        # Combine real and fake data for discriminator training
        # This approach helps avoid the shape mismatch in gradients
        combined_data = tf.concat([real_data, generated_data], axis=0)
        combined_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
            axis=0
        )

        # Add some noise to the labels for better training
        combined_labels += 0.05 * tf.random.uniform(tf.shape(combined_labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data, training=True)
            disc_loss = self.cross_entropy(combined_labels, predictions)

        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_data = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.discriminator(generated_data, training=True)
            gen_loss = self.cross_entropy(tf.ones_like(fake_predictions), fake_predictions)

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}


# Main execution code
def run_qgan():
    # Parameters
    latent_dim = 2  # Dimension of latent space
    num_qubits = 3  # Number of qubits, adjustable based on needs
    epochs = 1000

    # Individual data in dictionary list format
    individuos = [
        {'n_estimators': 100, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 125, 'max_depth': 4, 'objective_function_values': 0.7303370786516854},
        {'n_estimators': 175, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 125, 'max_depth': 4, 'objective_function_values': 0.7303370786516854},
        {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 125, 'max_depth': 2, 'objective_function_values': 0.7528089887640449},
        {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 100, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 150, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 150, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 150, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 175, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 125, 'max_depth': 2, 'objective_function_values': 0.7528089887640449},
        {'n_estimators': 125, 'max_depth': 4, 'objective_function_values': 0.7303370786516854},
        {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 100, 'max_depth': 4, 'objective_function_values': 0.7191011235955056},
        {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 125, 'max_depth': 2, 'objective_function_values': 0.7528089887640449},
        {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 125, 'max_depth': 2, 'objective_function_values': 0.7528089887640449},
        {'n_estimators': 100, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 125, 'max_depth': 4, 'objective_function_values': 0.7303370786516854},
        {'n_estimators': 175, 'max_depth': 2, 'objective_function_values': 0.7415730337078652},
        {'n_estimators': 150, 'max_depth': 4, 'objective_function_values': 0.7191011235955056}
    ]

    # Normalization of data (min-max scaling)
    n_estimators_values = np.array([ind['n_estimators'] for ind in individuos], dtype=np.float32)
    max_depth_values = np.array([ind['max_depth'] for ind in individuos], dtype=np.float32)
    objective_values = np.array([ind['objective_function_values'] for ind in individuos], dtype=np.float32)

    # Normalize each feature
    n_estimators_values = (n_estimators_values - np.min(n_estimators_values)) / (
                np.max(n_estimators_values) - np.min(n_estimators_values))
    max_depth_values = (max_depth_values - np.min(max_depth_values)) / (
                np.max(max_depth_values) - np.min(max_depth_values))
    objective_values = (objective_values - np.min(objective_values)) / (
                np.max(objective_values) - np.min(objective_values))

    # Stack to form X_real
    X_real = np.column_stack((n_estimators_values, max_depth_values, objective_values))

    # Convert to TensorFlow tensor
    X_real = tf.convert_to_tensor(X_real, dtype=tf.float32)

    # Show first normalized values
    print("First 5 normalized values:", X_real[:5])  # Debugging line

    # Create models
    generator = build_generator(latent_dim, output_dim=3)  # Match discriminator's input dimension
    discriminator = build_discriminator(input_dim=3)  # 3 features in our data

    # Create and compile QGAN
    qgan = QGAN(generator, discriminator, latent_dim)
    qgan.compile(
        gen_optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Lower learning rate
        disc_optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    # Train the quantum GAN with a smaller batch size
    print("Starting QGAN training...")  # Debugging line
    qgan.fit(X_real, epochs=epochs, batch_size=4)

    return qgan


if __name__ == "__main__":
    qgan = run_qgan()
