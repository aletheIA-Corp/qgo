# -- TODO: objeto de conexion a máquinas reales y logica de simulador
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

class QuantumSimulator:

    def __init__(self, technology: str = "aer"):

        """
        :param technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la logica.
        """

        self.technology: str = technology
        self.sampler = None

        match self.technology:

            case "aer":
                self.sampler = Sampler(AerSimulator())

        print(self.sampler)

    def run(self, qc, shots: int):

        """
        Metodo para ejecutar un simulador cuantico y obtener sus resultados
        :param qc: Circuito cuantico que se quiere medir
        :param shots: Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # -- Definimos el sampler para ejecutar shots cantidad de veces el circuito cuantico especificado
        job = self.sampler.run([qc], shots=shots)

        # -- Lanzamos el job (tarea de ejecución del circuito cuántico) y obtenemos sus resultados
        results = job.result()

        # -- Accedemos a los valores de las mediciones del circuito cuantico
        qc_results = results[0].data.c

        # -- Contamos la probabilidad de los resultados
        qc_results = qc_results.get_counts()

        return qc_results



class QuantumMachine:

    def __init__(self):
        pass

    def run(self):
        pass


class QuantumTechnology:

    def __init__(self, quantum_technology: str = "simulator", technology: str = "aer", qm_api_key: str | None = None, qm_connection_service: str | None = None,
                 quantum_machine: str = "least_busy"):

        """
        Metodo que instancia un objeto QuantumTechnology, el cual puede ser un simulador o un conector a una máquina cuántica real
        :param quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro technology. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha: ibm_quantum | ibm_cloud
        :param quantum_machine. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        """

        self.quantum_technology: str = quantum_technology
        self.technology: str = technology
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: str | None = qm_connection_service
        self.quantum_machine: str = quantum_machine

        # -- TODO: Diccionario de tecnologias y servicios habilitados (actualizar periódicamente)
        self._allowed_quantum_tech: dict = {"simulator": ["aer"],
                                            "quantum_machine":["ibm"],
                                            "quantum_machines": {"ibm": ["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]},
                                            "quantum_services": {"ibm": ["ibm_quantum", "ibm_cloud"]}}

        # -- Validamos que los parámetros relacionados con la tecnología cuántica son correctos
        self.validate_input_parameters()

        # -- Generamos el objeto que ejecuta el algoritmo cuántico
        self.execution_object: Simulator | QuantumMachineConnector | None = None

        match quantum_technology:

            case "simulator":

                self.execution_object: Simulator = QuantumSimulator(self.technology)

            case "quantum_machine":

                self.execution_object: QuantumMachine = QuantumMachine()

        print("self.execution_object: ", self.execution_object)
        print("self.technology: ", self.technology)

    def get_quantum_technology(self):
        """
        Metodo getter para retorna el execution object (objeto que ejecuta un circuito en un ordenador cuántico o en un simulador cuántico)
        :return: self.execution_object
        """
        return self.execution_object

    def validate_input_parameters(self) -> bool:
        """
        Metodo para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # -- Validar la tecnologia cuántica definida
        if self.quantum_technology == "simulator":
            if self.technology not in self._allowed_quantum_tech["simulator"]:
                raise ValueError(f"self.quantum_technology: La randomness_technology escogida es {self.quantum_technology}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['simulator']}")

        if self.quantum_technology == "quantum_machine":

            if self.technology not in self._allowed_quantum_tech["quantum_machine"]:
                raise ValueError(f"self.technology: La technology escogida es {self.technology}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machine']}")

            if self.quantum_machine not in self._allowed_quantum_tech["quantum_machines"][f"{self.technology}"]:
                raise ValueError(f"self.quantum_machine: La quantum_machine escogida es {self.quantum_machine}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machines'][f'{self.technology}']}")

            if self.qm_connection_service not in self._allowed_quantum_tech["quantum_services"][f"{self.technology}"]:
                raise ValueError(f"self.qm_connection_service: El qm_connection_service escogido es {self.qm_connection_service}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_services'][f'{self.technology}']}")

        return True


    def quantum_random_real(self, min_value: int | float, max_value: int | float, num_qubits: int =14):
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
        # simulator = Aer.get_backend('qasm_simulator')
        result = self.execution_object.run(qc, 1)

        result = list(result.keys())[0]

        # Convertir de binario a decimal y normalizar en [0,1]
        random_decimal = int(result, 2) / (2 ** num_qubits)

        # Escalar el número aleatorio al rango deseado
        scaled_random = min_value + random_decimal * (max_value - min_value)

        print(scaled_random)

        # Redondear a 4 decimales
        return round(scaled_random, 4)