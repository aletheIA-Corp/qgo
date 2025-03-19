import math

from qiskit import QuantumCircuit

from quantum_technology import QuantumTechnology, QuantumSimulator, QuantumMachine
from genethic_individuals import Individual

from typing import Literal, Dict, Union, Tuple, List


class Generator:

    def __init__(self,
                 num_individuals: int,
                 bounds_dict: Dict[str, Tuple[Union[int, float]]],
                 child_values: List | None,
                 generation: int = 0,
                 max_qubits: int = 14,
                 operation: Literal["generate", "reproduct"] = "generate",
                 quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 quantum_service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy"):

        # -- Definimos la el tipo de tecnología y la el servicio a utilizar
        self.operation: Literal["generate", "reproduct"] = operation
        self.quantum_technology: Literal["simulator", "quantum_machine"] = quantum_technology
        self.quantum_service: Literal["aer", "ibm"] = quantum_service
        self.qm_api_key: str = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine

        # -- Definimos variables de los metodos
        self.op_executor: QuantumMachine | QuantumSimulator | None = None

        # -- Generamos aleatoriamente los individuos
        self.num_individuals: int = num_individuals
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.child_values: List | None = child_values
        self.generation: int = generation
        self.max_qubits: int = max_qubits

        # -- Creo la propiedad de valores del individuo
        self.individual_values: Dict[str, Union[int, float]] = {}

        # -- Creamos los individuos y los almacenamos en una lista
        # self.individuals_list: List[Individual] = []
        # for i in range(self.num_individuals):
            # self.individuals_list.append(Individual(self.randomness_executor, self.bounds_dict, None, 14))
        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores

    def generate_individuals(self):
        individual_list: List[Individual] = []

        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores
        if self.child_values is None:

            for parameter, v in self.bounds_dict.items():
                self.individual_values = Individual(self.bounds_dict, self.child_values, self.max_qubits, self.generation)
                self.individual_values[parameter] = self.generate_random_value((v["limits"][0], v["limits"][1]),
                                                                               v["type"])
                individual_list.append(self.individual_values)

        else:
            for parameter, cv in zip([z for z in self.bounds_dict.keys()], self.child_values):
                self.individual_values[parameter] = cv




        if self.operation == "generate":
            self.op_executor = self.operation_executor()
            self.op_executor.run()

        elif self.operation == "reproduct":
            pass

        return individual_list

    def operation_executor(self):

        # -- Creamos los ejecutores cuánticos para la aletoriedad y el algoritmo de optimizacion
        self.op_executor: QuantumMachine | QuantumSimulator | None = QuantumTechnology(self.quantum_technology,
                                                                        self.quantum_service,
                                                                        self.qm_api_key,
                                                                        self.qm_connection_service,
                                                                        self.quantum_machine).get_quantum_technology()

        return self.op_executor

    def generate_random_value(self, val_tuple: tuple, data_type: str, max_qubits: int):
        if data_type == "int":
            return int(self.quantum_random_real(val_tuple[0], val_tuple[1], math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1))))

        elif data_type == "float":
            dynamic_max_qubits = max_qubits
            if math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1)) > max_qubits:
                dynamic_max_qubits = int(math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1)) + 4)
                raise Warning(f"El numero maximo de qubits estipulado es {max_qubits}, pero para representar el numero {(max(val_tuple[0], val_tuple[1]))} se necesitan minimo para la parte natural {math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1))} qubits.\n Se corrige dinámicamente para que tenga {dynamic_max_qubits} digitos decimales.")
            return self.quantum_random_real(val_tuple[0], val_tuple[1], dynamic_max_qubits)

    def quantum_random_real(self, min_value: int | float, max_value: int | float, num_qubits: int = 14):
        """
        Genera un número aleatorio cuántico entre min_value y max_value.

        Parámetros:
        - min_value: Límite mínimo del rango
        - max_value: Límite máximo del rango
        - num_qubits: Número de qubits para la generación (por defecto 14)

        Retorna:
        Un número aleatorio entre min_value y max_value
        """

        # -- Creamos el circuito cuántico
        qc = QuantumCircuit(num_qubits, num_qubits)

        # -- Aplicar Hadamard a todos los qubits para lograr una superposición uniforme
        qc.h(range(num_qubits))

        # -- Medimos todos los qubits
        qc.measure(range(num_qubits), range(num_qubits))

        # -- Ejecutamos el circuito
        result = self.execution_object.run(qc, 1)

        # -- Obtenemos los resultados
        result = list(result.keys())[0]

        # -- Convertimos el numero binario a decimal y lo normalizamo entre [0,1]
        random_decimal = int(result, 2) / (2 ** num_qubits)

        # -- Obtenemos el numero cuántico aleaotorio buscado
        return min_value + random_decimal * (max_value - min_value)

