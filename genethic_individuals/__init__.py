import random
from typing import Dict, Union, Tuple, List


class Individual:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], child_values: List | None):
        """
        Clase que va a instanciar los distintos individuos que van a competir.
        :param bounds_dict: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación.
        """

        # -- Almaceno parámetros en propiedades
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict

        # -- Almaceno los valores que provienen de la generacion del individuo (sus valores reales)
        self.child_values: List | None = child_values

        # -- Creo la propiedad de valores del individuo
        self._individual_values: Dict[str, Union[int, float]] = {}

        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores
        if child_values is None:

            for parameter, v in self.bounds_dict.items():
                self._individual_values[parameter] = self.generate_random_value((v["limits"][0], v["limits"][1]), v["type"])
        else:
            for parameter, cv in zip([z for z in self.bounds_dict.keys()], child_values):
                self._individual_values[parameter] = cv

        # -- Almaceno en una propiedad si el individuo tiene una malformación
        self.malformation: bool = self.exists_malformation()

    def exists_malformation(self) -> bool:
        """
        Método para saber si el individuo tiene valores fuera del rango
        :return: True si existe malformacion, False else
        """

        for k, v in self._individual_values.items():
            individual_value: int | float = self._individual_values[k]
            individual_restrictions: tuple = self.bounds_dict[k]["malformation_limits"]

            if individual_value < min(individual_restrictions) or individual_value > max(individual_restrictions):
                return True

        return False

    def get_individual_values(self) -> List[Union[int, float]]:
        """
        Método que va a devolver los valores del individuo en una lista. por ejemplo, si viene asi: {learning_rate: 0.0125, batch_size: 34}
        :return:
        """
        return self._individual_values

    @staticmethod
    def generate_random_value(val_tuple: tuple, data_type: str):
        if data_type == "int":
            return random.randint(val_tuple[0], val_tuple[1])
        elif data_type == "float":
            return random.uniform(val_tuple[0], val_tuple[1])
