import sys

from genethic_tournament_methods.qgan_reproductor import QGANReproductor
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

            # -- Si existe malformation_limits en la variable...
            if "malformation_limits" in [z for z in self.bounds_dict.keys()]:

                # -- Obtenemos el valor mínimo y máximo de malformaciones
                individual_restrictions: tuple = self.bounds_dict[k]["malformation_limits"]

                # -- Revisamos si el valor actual del individuo es inferior o superior al dict de malformaciones
                if individual_value[k] < min(individual_restrictions) or individual_value[k] > max(individual_restrictions):
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

    def __eq__(self, other, decimals=4):
        """
        Compara si dos individuos son iguales en base a sus propiedades con precisión decimal.
        :param other: Otro objeto de la clase Individual con el que se realizará la comparación.
        :param decimals: Número de decimales a considerar en la comparación (por defecto 4).
        :return: True si ambos individuos tienen las mismas propiedades con la precisión dada, False en caso contrario.
        """

        # Verificar que el otro objeto es de la clase Individual
        if not isinstance(other, Individual):
            return False

        # Obtener los valores de los individuos
        values_self = self.get_individual_values()
        values_other = other.get_individual_values()

        # Redondear los valores antes de compararlos
        rounded_self = {k: round(v, decimals) if isinstance(v, float) else v for k, v in values_self.items()}
        rounded_other = {k: round(v, decimals) if isinstance(v, float) else v for k, v in values_other.items()}

        return rounded_self == rounded_other

    def set_individual_value(self, parameter: str, new_value: float | int):
        self._properties[parameter] = new_value
        self._malformation = self.exists_malformation()


class Population:

    def __init__(self):

        """
        Constructor de la clase population constituida por Individuals.
        """
        # -- Definimos la variable que contendrá las poblaciones distinguidas por generacion
        self._population: Dict[str, List[Individual]] = {"0": []}

    def get_individuals(self, generation: Union[int, None] = None) -> Union[
        Dict[str, List[Individual]], List[Individual]]:
        """
        Metodo getter para obtener los individuos de la población.

        :param generation: (int | None) Generación de la que se quiere obtener los individuos (None trae todas).
        :return: Diccionario completo si generation es None, o la lista de individuos de la generación elegida.
        """

        # -- Si no hemos pasado ningún valor a generation...
        if generation is None:

            # -- Devolvemos el diccionario completo
            return self._population

        # -- Si hemos pasado un valor, entonces obtenemos la generación deseada (o lista vacía en su defecto)
        return self._population.get(str(generation), [])  # Devuelve la generación o una lista vacía si no existe

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
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy",
                 individuals_to_reproduct: List[Individual] | None = None,
                 reproductor: Literal["QGAN"] | None = None):

        """
        Metodo para poblar una población con individuos de la clase Individual
        :param generation: (int) Generación que se quiere poblar
        :param num_individuals: (int) Numero de individuos total de la población
        :param bounds_dict: (Dict) Diccionario de bounds con limites normales y malformaciones
        :param max_qubits: (int) Número máximo de qubits que se utilizarán para la parte cuántica
        :param operation: (Literal["generate", "reproduct"]) Operación que se quiere realizar
        :param quantum_technology: (Literal["simulator", "quantum_machine"]) Tecnología cuántica a utilizar
        :param quantum_service: (Literal["aer", "ibm"]) Servicio cuántico a utilizar
        :param qm_api_key: (str) API key para utilizar los ordenadores cuánticos
        :param qm_connection_service: (Literal["ibm_quantum", "ibm_cloud"] | None) Servicio de conexión
        :param quantum_machine: (Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]) Maquina cuántica que ejecuta el circuito cuántico
        :param individuals_to_reproduct: (List[Individual] | None) Individuos a reproductir (operation == "reproduct")
        :param reproductor: (Literal["QGAN"] | None) Tipo de reproductor que se utilizará para generar nuevos hijos
        """

        match operation:

            case "generate":

                # -- Generamos las propiedades de los individuos por medio de circuitos cuánticos
                individuals_properties: dict = Generator(operation=operation,
                                                         num_individuals=num_individuals,
                                                         bounds_dict=bounds_dict,
                                                         max_qubits=max_qubits,
                                                         quantum_technology=quantum_technology,
                                                         quantum_service=quantum_service,
                                                         qm_api_key=qm_api_key,
                                                         qm_connection_service=qm_connection_service,
                                                         quantum_machine=quantum_machine,
                                                         reproductor=reproductor).generate_properties()

            case "reproduct":

                if individuals_to_reproduct is None:
                    sys.exit(f"La operación {operation} requiere List[Individual] -> Ver: individuals_to_reproduct=")

                # -- Instanciamos el generador de datos cuánticos para casos de reproducción
                reproduct_generator: Generator = Generator(operation=operation,
                                                           num_individuals=num_individuals,
                                                           bounds_dict=bounds_dict,
                                                           max_qubits=max_qubits,
                                                           quantum_technology=quantum_technology,
                                                           quantum_service=quantum_service,
                                                           qm_api_key=qm_api_key,
                                                           qm_connection_service=qm_connection_service,
                                                           quantum_machine=quantum_machine,
                                                           reproductor=reproductor)

                # -- Generamos las propiedades de los individuos por medio de circuitos cuánticos
                individuals_properties = reproduct_generator.reproduct_properties(individuals=individuals_to_reproduct,
                                                                                  samples=num_individuals,
                                                                                  epochs=300,
                                                                                  generation=generation)

            case _:
                sys.exit(f"El generador no admite la operacion {operation} (utilizar: 'generate' | 'reproduct')")


        # -- Revisamos que no exista la generación (si no existe creamos una nueva clave y lista de individuos)
        if str(generation) not in [z for z in self._population.keys()]:
            self._population[str(generation)] = []

        # -- Iteramos sobre las propiedades creadas con los circuitos cuánticos
        for properties in individuals_properties.values():

            # -- Creamos individuos con las propiedades calculadas con los circuitos cuánticos
            individual = Individual(bounds_dict=bounds_dict, properties=properties, generation=generation)

            # -- Obtenemos los valores del individuo creado
            individual_values: dict = individual.get_individual_values()

            match operation:

                case "generate":

                    # -- Si el individuo no tiene malformaciones...
                    if not individual_values.get("malformation"):

                        # -- Pasamos a comprobar si ya existe un individuo idéntico en la lista de esta generacion...
                        is_duplicate = any(existing_indv == individual for existing_indv in self.get_individuals(generation))

                        # -- Lo agregamos como parte de la población de la generación
                        if not is_duplicate:
                            self.get_individuals(generation).append(individual)

                case "reproduct":
                    self.get_individuals(generation).append(individual)

    def print_population(self, generation: int | None = None):
        """
        Metodo para imprimir la población en consola.
        :param generation: (int | None) Generación específica a imprimir (int) o todas si es None.
        """

        # -- Determinamos la población de qué generación de individuos queremos imprimir
        populations = self._population if generation is None else {generation: self._population.get(str(generation), [])}
        print(populations)

        for gen_id, individuals in populations.items():
            print("\n" + "#" * 90)
            print(f"Individuos de la generación: {gen_id}")
            print("#" * 90 + "\n")

            for idx, individual in enumerate(individuals):
                print(f"Individuo_{idx}: {individual.get_individual_values()}")

            print("\n" + "#" * 90)






