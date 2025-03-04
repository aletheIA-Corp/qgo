class QuantumMethods:

    @staticmethod
    def quantum_random_real(min_value, max_value, num_qubits=14):
        """
        Genera un número aleatorio cuántico entre min_value y max_value.

        Parámetros:
        - min_value: Límite mínimo del rango
        - max_value: Límite máximo del rango
        - num_qubits: Número de qubits para la generación (por defecto 14)

        Retorna:
        Un número aleatorio entre min_value y max_value
        """
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Aplicar Hadamard a todos los qubits para superposición uniforme
        qc.h(range(num_qubits))

        # Medir todos los qubits
        qc.measure(range(num_qubits), range(num_qubits))

        # Simulación
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(qc, simulator, shots=1).result()
        counts = result.get_counts()

        # Obtener el único resultado de medición
        bitstring = list(counts.keys())[0]

        # Convertir de binario a decimal y normalizar en [0,1]
        random_decimal = int(bitstring, 2) / (2 ** num_qubits)

        # Escalar el número aleatorio al rango deseado
        scaled_random = min_value + random_decimal * (max_value - min_value)

        # Redondear a 4 decimales
        return round(scaled_random, 4)

    # Generar un número cuántico real aleatorio con precisión 0.0001
    print(quantum_random_real(14))


