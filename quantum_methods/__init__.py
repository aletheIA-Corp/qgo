import math
import sys

from qiskit import QuantumCircuit

from quantum_technology import QuantumTechnology, QuantumSimulator, QuantumMachine
from genethic_individuals import Individual

from typing import Literal, Dict, Union, Tuple, List


class Generator:

    def __init__(self,
                 num_individuals: int,
                 bounds_dict: Dict[str, Tuple[Union[int, float]]],
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

        # -- Definimos el ejecutor de operaciones en el simulador u ordenador cuántico
        self.executor: QuantumMachine | QuantumSimulator | None = None

        # -- Definimos la cantidad, el diccionario de características a generar/reproducir y los qubits a utilizar
        self.num_individuals: int = num_individuals
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.max_qubits: int = max_qubits

        # -- Creo la propiedad de valores del individuo
        self.individual_values: Dict[str, Union[int, float]] = {}

        # -- Guardamos el numero de qubits utilizado para generar el circuito de cada propiedad de los individuos
        self.indv_prop_num_qubits: dict = {}

    """def generate_individuals(self):
        future_indv_params: List[dict] = []
        individual_list: List[Individual] = []

        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores
        if self.child_values is None:

            # -- TODO: primero necesitamos obtener los numeros
            for individual in range(0, self.num_individuals):
                future_params: Dict[str: str | int | float] = {}
                for parameter, v in self.bounds_dict.items():
                    future_params[parameter] = self.generate_random_value((v["limits"][0], v["limits"][1]), v["type"])
                future_indv_params.append(future_params)

        # -- TODO: luego crear los individuos
            for individual in range(0, self.num_individuals):
                self.individual_values = Individual(self.bounds_dict, self.child_values, self.max_qubits, self.generation)
                individual_list.append(self.individual_values)

        else:
            for parameter, cv in zip([z for z in self.bounds_dict.keys()], self.child_values):
                self.individual_values[parameter] = cv




        if self.operation == "generate":
            self.executor = self.operation_executor()
            self.executor.run()

        elif self.operation == "reproduct":
            pass

        return individual_list"""

    def generate_individuals(self):

        # -- Generamos la lista de circuitos cuánticos por caracteristica de los individuos para cada individuo
        qc_list: List[QuantumCircuit] = []

        # -- Para cada individuo que se debe generar
        for individual in range(0, self.num_individuals):
            self.indv_prop_num_qubits[individual] = {}

            # -- Para cada parámetro de los individuos a generar
            for parameter in self.bounds_dict.keys():

                if ("int" or "Int") in type(self.bounds_dict[parameter][0]):
                    dynamic_max_qubits = math.ceil(math.log2(len(str(max(self.bounds_dict[parameter][0], self.bounds_dict[parameter][1]))) + 1))

                elif ("floar" or "Float") in type(self.bounds_dict[parameter][0]):
                    dynamic_max_qubits = self.max_qubits
                    if math.ceil(math.log2(len(str(max(self.bounds_dict[parameter][0], self.bounds_dict[parameter][1]))) + 1)) > self.max_qubits:
                        dynamic_max_qubits = int(math.ceil(math.log2(len(str(max(self.bounds_dict[parameter][0], self.bounds_dict[parameter][1]))) + 1)) + 4)
                        raise Warning(f"El numero maximo de qubits estipulado es {self.max_qubits}, pero para representar el numero {(max(self.bounds_dict[parameter][0], self.bounds_dict[parameter][1]))} se necesitan minimo para la parte natural {math.ceil(math.log2(len(str(max(self.bounds_dict[parameter][0], self.bounds_dict[parameter][1]))) + 1))} qubits.\n Se corrige dinámicamente para que tenga {dynamic_max_qubits} digitos decimales.")
                else:
                    sys.exit("Se está intentando calcular el numero de qubits necesarios a partir de un valor no numerico")

                self.indv_prop_num_qubits[individual][parameter] = dynamic_max_qubits
                temp_qc: QuantumCircuit = self.generate_qc(dynamic_max_qubits)
                qc_list.append(temp_qc)

        # -- Ejecutamos todos los circuitos cuánticos bajo una misma sesion
        results: List[bytes] = self.quantum_random_real(qc_list)

        # -- Generamos un diccionario de resultados para adjudicar de forma ordenada a cada parametro de cada individuo
        results_dict: dict = {}

        # -- Para cada individuo que se debe generar
        for individual in range(0, self.num_individuals):
            results_dict[individual] = {}

            # -- Para cada parámetro de los individuos a generar a partir de los bytes binarios
            for parameter in self.bounds_dict.keys():
                results_dict[individual][parameter] = self.calculate_random_values(results[individual],
                                                                                   self.bounds_dict[parameter][0],
                                                                                   self.bounds_dict[parameter][1],
                                                                                   self.indv_prop_num_qubits[individual][parameter])

        return results_dict, self.indv_prop_num_qubits

    def operation_executor(self):

        # -- Creamos el ejecutor de operaciones cuánticas para la aletoriedad y el algoritmo de optimizacion
        self.executor = QuantumTechnology(self.quantum_technology,
                                          self.quantum_service,
                                          self.qm_api_key,
                                          self.qm_connection_service,
                                          self.quantum_machine).get_quantum_technology()

        return self.executor

    @staticmethod
    def generate_qc(max_qubits: int) -> QuantumCircuit:

        # -- Creamos el circuito cuántico
        qc = QuantumCircuit(max_qubits, max_qubits)

        # -- Aplicar Hadamard a todos los qubits para lograr una superposición uniforme
        qc.h(range(max_qubits))

        # -- Medimos todos los qubits
        qc.measure(range(max_qubits), range(max_qubits))

        return qc


    def quantum_random_real(self, qcs: List[QuantumCircuit]) -> List[bytes]:
        """
        Genera un número aleatorio cuántico entre min_value y max_value.

        Parámetros:
        - min_value: Límite mínimo del rango
        - max_value: Límite máximo del rango
        - num_qubits: Número de qubits para la generación (por defecto 14)

        Retorna:
        Un número aleatorio entre min_value y max_value
        """

        # -- Ejecutamos el circuito
        result = self.executor.run(qcs, 1)

        # -- TODO: creo que hay que quitar el [0]
        # -- Obtenemos los resultados
        # result = list(result.keys())[0]
        result = list(result.keys())

        return result

    @staticmethod
    def calculate_random_values(result, min_value: int | float, max_value: int | float, num_qubits: int = 14):

        # -- Convertimos el numero binario a decimal y lo normalizamo entre [0,1]
        random_decimal = int(result, 2) / (2 ** num_qubits)

        # -- Obtenemos el numero cuántico aleaotorio buscado
        return min_value + random_decimal * (max_value - min_value)

