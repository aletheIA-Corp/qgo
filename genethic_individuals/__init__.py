from quantum_methods import Generator

from typing import Dict, Union, Tuple, List, Literal


class Individual:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], properties: Dict[str, Union[int, float]], generation: int):
        """
        Inicializa un individuo con sus propiedades, restricciones y generación.

        :param bounds_dict: Dict que define los parámetros a optimizar, sus valores límite aceptables y de malformación.
        :param properties: Dict con los valores específicos de los parámetros para este individuo.
        :param generation: Número entero que indica la generación a la que pertenece el individuo.
        """

        # -- Obtenemos el bound_dict con los valores estándares mínimos y máximos de cada propiedad para cada individuo
        self.bounds_dict: Dict = bounds_dict

        # -- Obtenemos los valores que provienen de la generacion cuántica para este individuo
        self._properties: Dict[str, Union[int, float]] = properties

        # -- Obtenemos la propiedad generacion que refleja la generación en la que estamos iterando
        self._generation: int = generation

        # -- Determinamos si el individuo presenta una malformación según sus restricciones
        self._malformation: bool = self.exists_malformation()

        # -- Determinamos el valor de la función objetivo del individuo
        self._objective_function_values: float | None = None

    def exists_malformation(self) -> bool:
        """
        Verifica si el individuo tiene valores fuera del rango permitido.

        :return: True si el individuo presenta una malformación (valores fuera de los límites), False en caso contrario.
        """

        # -- Iteramos por las propiedades generadas con los circuitos cuánticos
        for k, v in self._properties.items():

            # -- Obtenemos el valor del individuo para cada propiedad
            individual_value: int | float = self._properties[k]

            # -- Obtenemos el valor mínimo y máximo de malformaciones
            individual_restrictions: tuple = self.bounds_dict[k]["malformation_limits"]

            # -- Revisamos si el valor actual del individuo es inferior o superior al dict de malformaciones
            if individual_value < min(individual_restrictions) or individual_value > max(individual_restrictions):
                return True

        return False

    def add_or_update_variable(self, var_name: str, value: int | float) -> None:
        """
        Agrega o actualiza una variable de instancia en el objeto Individual.

        :param var_name: Nombre de la variable de instancia.
        :param value: Valor de la variable (puede ser de cualquier tipo).
        """
        setattr(self, f"_{var_name}", value)

    def get_individual_values(self):

        """
        Retorna un diccionario con los valores del individuo, incluyendo la generación y el estado de malformación.

        :return: Diccionario con las propiedades del individuo, su generación y si presenta malformaciones insalvables
        """

        return ({k: v for k, v in self._properties.items()} | {"generation": self._generation} |
                {"malformation": self._malformation} | {"objective_function_values": self._objective_function_values})

    def __eq__(self, other):

        """
        Compara si dos individuos son iguales en base a sus propiedades.
        :param other: Otro objeto de la clase Individual con el que se realizará la comparación.

        :return: True si ambos individuos tienen las mismas propiedades, generación y estado de malformación; False en caso contrario.
        """

        # -- Revisamos que ambos individuos sean de la clase Individual
        if not isinstance(other, Individual):
            return False

        # -- Comparamos los valores entre ambos individuos
        return self.get_individual_values() == other.get_individual_values()


class Population:

    def __init__(self):

        """
        Constructor de la clase population constituida por Individuals.
        """
        # -- Definimos la variable que contendrá las poblaciones distinguidas por generacion
        self._population: Dict[str, List[Individual]] = {}


    def get_individuals(self, generation: int | None = None) -> List[Individual] | Dict[str, List[Individual]]:
        """
        Metodo getter para los individuos de la población
        :param generation: (int | None) Generación de la que se prentede los indiviudos (None trae todas las gen)
        :return: Población de la generación elegida o todas las generaciones y sus poblaciones (generation=None)
        """
        return self._population[str(generation)]

    def populate(self,
                 generation: int,
                 num_individuals: int,
                 bounds_dict: Dict,
                 max_qubits: int = 14,
                 operation: Literal["generate", "reproduct"] = "generate",
                 quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 quantum_service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy"):

        # -- Generamos las propiedades de los individuos por medio de circuitos cuánticos
        individuals_properties = Generator(operation=operation,
                                           num_individuals=num_individuals,
                                           bounds_dict=bounds_dict,
                                           max_qubits=max_qubits,
                                           quantum_technology=quantum_technology,
                                           quantum_service=quantum_service,
                                           qm_api_key=qm_api_key,
                                           qm_connection_service=qm_connection_service,
                                           quantum_machine=quantum_machine).generate_properties()

        # -- Revisamos que no exista la generación (si no existe creamos una nueva clave y lista de individuos)
        if str(generation) not in [z for z in self._population.keys()]:
            self._population[str(generation)] = []

        # -- Iteramos sobre las propiedades creadas con los circuitos cuánticos
        for properties in individuals_properties.values():

            # -- Creamos individuos con las propiedades calculadas con los circuitos cuánticos
            individual = Individual(bounds_dict=bounds_dict, properties=properties, generation=generation)

            # -- Obtenemos los valores del individuo creado
            individual_values: dict = individual.get_individual_values()

            # -- Si el individuo no tiene malformaciones...
            if not individual_values.get("malformation"):

                # -- Pasamos a comprobar si ya existe un individuo idéntico en la lista de esta generacion... y si no...
                is_duplicate = any(existing_indv == individual for existing_indv in self.get_individuals(generation))

                # -- Lo agregamos como parte de la población de la generación
                if not is_duplicate:
                    self.get_individuals(generation).append(individual)

    """def add_individual(self, individual: Individual, generation: int):

        # -- Revisamos que no exista la generación (si no existe creamos una nueva clave y lista de individuos)
        if generation not in self.get_individuals():
            self.get_individuals()[generation] = []  # Inicializar lista si no existe la generación

        # -- Verificamos si el individuo tiene algún valor de malformación
        individual_values: dict = individual.get_individual_values()

        # -- Si el individuo no tiene malformaciones
        if not individual_values.get("malformation"):

            # -- Comprobamos si ya existe un individuo idéntico en la lista de esta generacion
            is_duplicate = any(existing_individual == individual for existing_individual in self.get_individuals()[generation])
            print(individual_values)
            print(is_duplicate)

            # -- Si el individuo no tiene malformaciones insalvables y no es un duplicado lo agregamos a la población
            if not is_duplicate:
                self.get_individuals()[generation].append(individual)"""

    @staticmethod
    def force_repopulation(bounds_dict: Dict, individual_model: Individual, mutation_rate: float = 1.2,
                           generation: int = 0, max_attempts: int = 10) -> Individual:
        """
        Metodo para forzar la repoblación en caso de que no se alcance el num_individuals.
        :param bounds_dict: (Dict) Define los parámetros a optimizar, sus valores límite aceptables y de malformación.
        :param individual_model: (Individual) Individuo que se utilizará para rellenar forzosamente la población.
        :param mutation_rate: (float) Tasa de mutación inicial.
        :param generation: (int) Indica la generación a la que pertenece el individuo.
        :param max_attempts: (int) Número máximo de intentos para encontrar un individuo válido.
        :return: Un nuevo individuo de la clase Individual.
        """

        attempts = 0
        found_valid_individual = False
        properties: dict = {}
        new_individual: Individual | None = None

        while attempts < max_attempts and not found_valid_individual:

            # -- Intentamos mutar el individuo con la tasa de mutación actual
            properties.clear()  # Limpiar propiedades en cada intento nuevo

            # Mutar atributos del individuo
            for attr, value in individual_model.get_individual_values().items():
                if attr != "malformation" and attr != "objective_function_values" and attr != "generation":
                    properties[attr] = int(value * mutation_rate) if isinstance(value, int) else (value * mutation_rate)

            # Crear un nuevo individuo con la tasa de mutación actual
            new_individual = Individual(bounds_dict, properties, generation)

            # -- Verificar si el nuevo individuo cumple los criterios (por ejemplo, que no esté malformado)
            if not new_individual.get_individual_values().get("malformation"):
                found_valid_individual = True  # Si es válido, detenemos la búsqueda
            else:
                # Incrementamos la tasa de mutación para el siguiente intento
                mutation_rate += 0.001  # Puedes ajustar este incremento según lo desees
                attempts += 1

        # Si no se encontró un individuo válido después de max_attempts, se devuelve el individuo aunque esté malformado
        if not found_valid_individual:
            print(
                f"No se encontró un individuo válido después de {max_attempts} intentos. Se devolverá el último individuo.")
            # No verificamos si está malformado, simplemente devolvemos el individuo
            new_individual = individual_model

        return new_individual

    def print_population(self, generation: int | None = None):
        """
        Metodo para imprimir la población en consola.
        :param generation: (int | None) Generación específica a imprimir (int) o todas si es None.
        """

        # -- Determinamos la población de qué generación de individuos queremos imprimir
        populations = self._population if generation is None else {generation: self._population.get(str(generation), [])}

        for gen_id, individuals in populations.items():
            print("\n" + "#" * 90)
            print(f"Individuos de la generación: {gen_id}")
            print("#" * 90 + "\n")

            for idx, individual in enumerate(individuals):
                print(f"Individuo_{idx}: {individual.get_individual_values()}")

            print("\n" + "#" * 90)






