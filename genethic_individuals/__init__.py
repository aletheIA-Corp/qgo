import math
from os import getenv
from typing import Dict, Union, Tuple, List
from quantum_technology import QuantumTechnology


class Individual:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], properties: dict, generation: int):
        """
        Clase que va a instanciar los distintos individuos que van a competir.
        :param bounds_dict: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación.
        :param child_values: Diccionario en el que se definen los parámetros a optimizar y sus valores a partir de la primera generacion, ej. '{learning_rate: (0.0001, 0.1)}'
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación
        """

        # -- Definimos el bound_dict para malformaciones
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict

        # -- Almaceno los valores que provienen de la generacion del individuo
        self._properties: dict = properties

        # -- Creo la propiedad de valores del individuo
        self._individual_values: Dict[str, Union[int, float]] = {}

        # -- Creamos la propiedad generacion
        self._generation: int = generation

        # -- Almaceno en una propiedad si el individuo tiene una malformación
        self._malformation: bool = self.exists_malformation()

    def exists_malformation(self) -> bool:
        """
        Metodo para saber si el individuo tiene valores fuera del rango
        :return: True si existe malformacion, False else
        """

        for k, v in self._individual_values.items():
            individual_value: int | float = self._individual_values[k]
            individual_restrictions: tuple = self.bounds_dict[k]["malformation_limits"]

            if individual_value < min(individual_restrictions) or individual_value > max(individual_restrictions):
                return True

        return False

    def get_individual_values(self):
        return {k: v for k, v in self._properties.items()} | {"generation": self._generation} | {"malformation": self._malformation}