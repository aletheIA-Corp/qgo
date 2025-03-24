from typing import List, Dict
from genethic_individuals import Individual
import numpy as np


class Mutation:
    def __init__(self, bounds_dict: Dict, individual_list: List[Individual], mutate_probability: float):

        """
        Constructor de la clase Mutation con la que mutaremos los genes de los individuos
        :param bounds_dict: (Dict) Diccionario de bounds de los individuos.
        :param individual_list: (List[Individual]) Lista de individuos que mutaremos.
        :param mutate_probability: (float) Probabilidad de mutación
        """

        self.individual_list: List[Individual] = individual_list
        self.mutate_probability: float = mutate_probability
        self.bounds_dict = bounds_dict

    def run_mutation(self, verbose: bool = True) -> List[Individual]:

        """
        Metodo para ejecutar las mutaciones
        :param verbose: (bool) Variable para determinar si imprimimos en consola o no.
        :return: Lista de individuos mutados
        """

        # -- Evaluamos si hay que mutar y cómo mutar cada individuo de la lista
        for idx, individual in enumerate(self.individual_list):

            if verbose:
                print("\n------------------------------------------------")
                print(f'Individuo: {idx}')

            # -- Realizamos los cruces de cada gen
            for parameter, bound in self.bounds_dict.items():

                # -- Dependiendo del tipo de bound en cuestión...
                match bound['bound_type']:

                    case 'predefined':

                        if np.random.rand() < self.mutate_probability:

                            if verbose:
                                print(f'{parameter} original (predefined): ', individual.get_individual_values())

                            individual.set_individual_value(parameter, self.mutation_bit_flip(individual, parameter))

                            if verbose:
                                print(f'{parameter} mutado (predefined): ', individual.get_individual_values())

                        else:
                            if verbose:
                                print(f'{parameter} no muta (predefined): ', individual.get_individual_values())

                    case 'interval':

                        if np.random.rand() < self.mutate_probability:

                            if verbose:
                                print(f'{parameter} original (interval): ', individual.get_individual_values())

                            individual.set_individual_value(parameter, self.mutation_uniform(individual, parameter))

                            if verbose:
                                print(f'{parameter} mutado (interval): ', individual.get_individual_values())

                        else:

                            if verbose:
                                print(f'{parameter} no muta (interval): ', individual.get_individual_values())

            if verbose:
                print("\n------------------------------------------------")

        return self.individual_list

    def mutation_bit_flip(self, individual: Individual, parameter: str):
        """
        Metodo para mutar valores discreto


        :param individual: Indivudo que se quiere mutar alguno de sus genes
        :param parameter: Parámetro que se quiere modificar del indiviudo
        :return: Parámetro mutado.
        """

        possible_values = [z for z in self.bounds_dict[parameter]["malformation_limits"] if z != individual.get_individual_values()[parameter]]
        return float(np.random.choice(possible_values)) if self.bounds_dict[parameter]["type"] == "float" else int(np.random.choice(possible_values))

    def mutation_uniform(self, individual, parameter):
        """
        Realiza una mutación uniforme en valores enteros o reales.

        :param individual: Indivudo que se quiere mutar alguno de sus genes
        :param parameter: Parámetro que se quiere modificar del indiviudo

        :return: Parámetro mutado.
        """

        parameter_bounds: list = [z for z in self.bounds_dict[parameter]["malformation_limits"] if z != individual.get_individual_values()[parameter]]

        match self.bounds_dict[parameter]["type"]:
            case "float":
                return float(np.random.uniform(parameter_bounds[0], parameter_bounds[1]))
            case "int":
                return int(np.random.uniform(parameter_bounds[0], parameter_bounds[1]))

    def mutate_repeated_individuals(self, verbose: bool = True):
        """
        Identifica los individuos repetidos en la generación actual y los muta obligatoriamente.
        :param verbose: (bool) Variable para determinar si imprimimos en consola o no.
        """

        # -- Diccionario para guardar los individuos
        seen: Dict = {}

        # -- Evaluamos si hay repeticiones de algún individuo
        for idx, individual in enumerate(self.individual_list):

            # -- Generamos un conjunto de valores únicos del individuo para verificar repetidos
            individual_values = individual.get_individual_values()
            unique_values = tuple(sorted(individual_values.items()))  # Convertimos a tupla hashable

            # -- Verificamos si ya hemos visto este conjunto de valores
            if unique_values in seen:

                # -- Mutamos obligatoriamente al individuo repetido
                for parameter, bound in self.bounds_dict.items():

                    match bound['bound_type']:

                        case 'predefined':

                            if verbose:
                                print('Individuo repetido (predefined): ', individual.get_individual_values())
                            individual.set_individual_value(parameter, self.mutation_bit_flip(individual, parameter))
                            if verbose:
                                print('Individuo repetido (predefined): ', individual.get_individual_values())

                        case 'interval':
                            if verbose:
                                print("Individuo repetido (interval): ", individual.get_individual_values())
                            individual.set_individual_value(parameter, self.mutation_uniform(individual, parameter))
                            if verbose:
                                print("Individuo repetido (interval): ",individual.get_individual_values())

            # -- Guardamos el individuo usando la tupla como clave
            seen[unique_values] = individual


