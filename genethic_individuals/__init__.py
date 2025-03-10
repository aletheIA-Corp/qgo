import math
from typing import Dict, Union, Tuple, List
from quantum_technology import QuantumTechnology


class Individual:
    def __init__(self, qm_conn_obj: QuantumTechnology, bounds_dict: Dict[str, Tuple[Union[int, float]]], child_values: List | None, max_qubits: int = 14, generation: int = 0):
        """
        Clase que va a instanciar los distintos individuos que van a competir.
        :param qm_conn_obj: Objeto de conexión con el ordenador o el simulador cuántico
        :param bounds_dict: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación.
        :param child_values: Diccionario en el que se definen los parámetros a optimizar y sus valores a partir de la primera generacion, ej. '{learning_rate: (0.0001, 0.1)}'
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación
        """

        # -- Almaceno parámetros en propiedades
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict

        # -- Almaceno los valores que provienen de la generacion del individuo (sus valores reales)
        self.child_values: List | None = child_values

        # -- Creo la propiedad de valores del individuo
        self.individual_values: Dict[str, Union[int, float]] = {}

        # -- Almacenamos el objeto de conexion
        self.qm_conn_obj: QuantumTechnology = qm_conn_obj

        # -- Almacenamos el numero maximo de qubits para operar en el simulador/ordenador cuantico
        self.max_qubits: int = max_qubits

        # -- Definimos la generacion
        self.generation = generation

        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores
        if child_values is None:

            for parameter, v in self.bounds_dict.items():
                self.individual_values[parameter] = self.generate_random_value((v["limits"][0], v["limits"][1]), v["type"])

        else:
            for parameter, cv in zip([z for z in self.bounds_dict.keys()], child_values):
                self.individual_values[parameter] = cv

        # -- Almaceno en una propiedad si el individuo tiene una malformación
        self.malformation: bool = self.exists_malformation()

    def exists_malformation(self) -> bool:
        """
        Metodo para saber si el individuo tiene valores fuera del rango
        :return: True si existe malformacion, False else
        """

        for k, v in self.individual_values.items():
            individual_value: int | float = self.individual_values[k]
            individual_restrictions: tuple = self.bounds_dict[k]["malformation_limits"]

            if individual_value < min(individual_restrictions) or individual_value > max(individual_restrictions):
                return True

        return False

    def get_individual_values(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los valores del individuo en una lista. por ejemplo, si viene asi: {learning_rate: 0.0125, batch_size: 34}
        :return:
        """

        return self.individual_values | {"generation": self.generation}

    def generate_random_value(self, val_tuple: tuple, data_type: str):
        if data_type == "int":
            return int(self.qm_conn_obj.quantum_random_real(val_tuple[0], val_tuple[1], math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1))))

        elif data_type == "float":
            dynamic_max_qubits = self.max_qubits
            if math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1)) > self.max_qubits:
                dynamic_max_qubits = int(math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1)) + 4)
                raise Warning(f"El numero maximo de qubits estipulado es {self.max_qubits}, pero para representar el numero {(max(val_tuple[0], val_tuple[1]))} se necesitan minimo para la parte natural {math.ceil(math.log2(len(str(max(val_tuple[0], val_tuple[1]))) + 1))} qubits.\n Se corrige dinámicamente para que tenga {dynamic_max_qubits} digitos decimales.")
            return self.qm_conn_obj.quantum_random_real(val_tuple[0], val_tuple[1], dynamic_max_qubits)
