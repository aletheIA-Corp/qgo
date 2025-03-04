from typing import Dict, List, Union, Callable, Tuple
from genethic_tournament_methods import GenethicTournamentMethods
from bounds_creator import BoundCreator
from genethic_individuals import *
from quantum_technology import QuantumTechnology, QuantumSimulator, QuantumMachine

# from qiskit import QuantumCircuit, Aer, execute


class QGO:
    def __init__(self,
                 bounds_dict: Dict[str, Tuple[Union[int, float]]],
                 num_generations: int,
                 num_individuals: int,
                 objective_function: Callable,
                 problem_type: str = "minimize",
                 tournament_method: str = "easimple",
                 podium_size: int = 3,
                 reproduction_variability: float = 0.2,
                 mutate_probability: float = 0.25,
                 mutation_center_mean: float = 0.0,
                 mutation_size: float = 0.5,
                 randomness_quantum_technology: str = "simulator",
                 randomness_technology: str = "aer",
                 optimizer_quantum_technology: str = "simulator",
                 optimizer_technology: str = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: str | None = None,
                 quantum_machine: str = "least_busy",
                 ):
        """
        Clase-Objeto padre para crear un algoritmo genético cuántico basado en QAOA y generacion de aleatoriedad cuántica
        en lo respectivo a mutaciones y cruces reproductivos.
        :param bounds_dict: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        :param num_generations: Numero de generaciones que se van a ejecutar
        :param num_individuals: Numero de Individuos iniciales que se van a generar
        :param objective_function: Función objetivo que se va a emplear para puntuar a cada individuo (debe retornar un float)
        :param problem_type: [minimize, maximize] Seleccionar si se quiere minimizar o maximizar el resultado de la función objetivo. Por ejemplo si usamos un MAE es minimizar,
         un Accuracy sería maximizar.
        :param tournament_method: [easimple, .....] Elegir el tipo de torneo para seleccionar los individuos que se van a reproducir.
        :param podium_size: Cantidad de individuos de la muestra que van a competir para elegir al mejor. Por ejemplo, si el valor es 3, se escogen iterativamente 3 individuos
        al azar y se selecciona al mejor. Este proceso finaliza cuando ya no quedan más individuos y todos han sido seleccionados o deshechados.
        :param reproduction_variability: También conocido como Alpha. α∈[0,1]. Directamente proporcional a la potencial variablidad entre hijos y padres. Ej. Si Alpha=0, los genes
        de los hijos solo van a poder mutar en una interpolación entre los valores de los padres, cumpliendo la siguiente ecuación: λ∈[−α,1+α], para este caso λ∈[0,1]. Si Alpha=0.5
        λ∈[−0.5,1.5]. Esto se calculará posteriormente en una magnitud proporcional a los valores de los genes de los padres.
        :param mutate_probability:Tambien conocido como indpb ∈[0, 1]. Probabilidad de mutar que tiene cada gen. Una probabilidad de 0, implica que nunca hay mutación,
        una probabilidad de 1 implica que siempre hay mutacion.
        :param mutation_center_mean: μ donde μ∈R. Sesgo que aplicamos a la mutación para que el gen aumente o disminuya su valor. Cuando es 0, existe la misma probabilidad de
        mutar positiva y negativamente. Cuando > 0, aumenta proporcionalmente la probabilidad de mutar positivamente y viceversa. v=v0+N(μ,σ)
        :param mutation_size σ donde σ>0. Desviación estándar de la mutación, Si sigma es muy pequeño (ej. 0.01), las mutaciones serán mínimas, casi insignificantes.
        Si sigma es muy grande (ej. 10 o 100), las mutaciones pueden ser demasiado bruscas, afectando drásticamente la solución.
        - Mutaciones pequeñas, estables 0.1 - 0.5
        - Balance entre estabilidad y exploración 0.5 - 1.0
        - Exploración agresiva 1.5 - 3.0
        :param bounds_restrictions. Reestricciones límite que aplicar a cada parámetro (lógica de negocio o lógica de realidad matemática) cuyo exceso no tiene sentido en el caso
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación.
        :param randomness_quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la que calculan los valores aleatorios. Si es simulator, se hará con un simulador
         definido en el parámetro randomness_technology. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro randomness_technology.
        :param randomness_technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la selección aleatoria de variables.
        :param optimizer_quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la se ejecuta el optimizador. Si es simulator, se hará con un simulador definido
         en el parámetro optimizer_technology. Si es quantum_machine, el algoritmo se ejecuta en una máquina cuántica definida en el parámetro optimizer_technology.
        :param optimizer_technology. ["aer", "ibm", "d-wave", etc.]. El servicio tecnológico con el cual se ejecuta la optimización.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha: ibm_quantum | ibm_cloud
        :param quantum_machine. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        """
        # -- Almaceno propiedades
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.num_generations: int = num_generations
        self.num_individuals: int = num_individuals
        self.objective_function: Callable = objective_function
        self.problem_type: str = problem_type
        self.tournament_method: str = tournament_method
        self.podium_size: int = podium_size
        self.reproduction_variability: float = reproduction_variability
        self.mutate_probability: float = mutate_probability
        self.mutation_center_mean: float = mutation_center_mean
        self.mutation_size: float = mutation_size
        self.mutation_size: float = mutation_size
        self.randomness_quantum_technology: str = randomness_quantum_technology
        self.randomness_technology: str = randomness_technology
        self.optimizer_quantum_technology: str = optimizer_quantum_technology
        self.optimizer_technology: str = optimizer_technology
        self.qm_api_key: str | None = qm_api_key,
        self.qm_connection_service: str | None = qm_connection_service,
        self.quantum_machine: str = quantum_machine

        # -- Instancio la clase GenethicTournamentMethods en GTM
        self.GTM: GenethicTournamentMethods = GenethicTournamentMethods()

        # -- Validamos los inputs
        self.validate_input_parameters()

        # -- Creamos los ejecutores cuánticos para la aletoriedad y el algoritmo de optimizacion
        self.randomness_executor: QuantumTechnology = QuantumTechnology(self.randomness_quantum_technology,
                                                                                        self.randomness_technology,
                                                                                        self.qm_api_key,
                                                                                        self.qm_connection_service,
                                                                                        self.quantum_machine)

        self.optimizer_executor: QuantumTechnology = QuantumTechnology(self.optimizer_quantum_technology,
                                                                                        self.optimizer_technology,
                                                                                        self.qm_api_key,
                                                                                        self.qm_connection_service,
                                                                                        self.quantum_machine)

        print(self.randomness_executor.quantum_random_real(1, 100, 14))

        # -- Creamos los individuos y los almacenamos en una lista
        self.individuals_list: List[Individual] = []
        for i in range(self.num_individuals):
            self.individuals_list.append(Individual(self.bounds_dict, None))

        print(self.individuals_list)

        # -- Entramos a la parte genetica

        # -- Se supone que ya hemos obtenido los hijos
        child_list: List[List] = [
            [0.08910494983403751, 36.5],
            [0.07170969052131343, 71.5],
            [0.005021695515233724, 16.5],
            [0.015734475416848345, 856.5],
        ]
        self.individuals_list = [Individual(self.bounds_dict, child_vals) for child_vals in child_list]

        for i in self.individuals_list:
            print(f"Malformation: {i.malformation} - Values: {i.get_individual_values()}")

        # self.individuals_list = Individuals(self.bounds_dict, self.num_individuals, False, child_list).get_individuals()

        print(self.individuals_list)

    def validate_input_parameters(self) -> bool:
        """
        Método para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # -- Validar el bounds_dict
        if not all(isinstance(valor, (int, float)) for param_data in self.bounds_dict.values()
                   for key in ["limits", "malformation_limits"] if key in param_data for valor in param_data[key]):
            raise ValueError("bounds_dict: No todos los valores en bounds_dict son int o float.")

        # -- Validar Enteros num_generations, num_individuals, podium_size
        if not isinstance(self.num_generations, int):
            raise ValueError(f"self.num_generations: Debe ser un entero y su tipo es {type(self.num_generations)}")
        if not isinstance(self.num_individuals, int):
            raise ValueError(f"self.num_individuals: Debe ser un entero y su tipo es {type(self.num_individuals)}")
        if not isinstance(self.podium_size, int):
            raise ValueError(f"self.podium_size: Debe ser un entero y su tipo es {type(self.podium_size)}")

        # -- Validar Flotantes reproduction_variability, mutate_probability, mutation_center_mean, mutation_size
        if not isinstance(self.reproduction_variability, float):
            raise ValueError(f"self.reproduction_variability: Debe ser un float y su tipo es {type(self.reproduction_variability)}")
        if not isinstance(self.mutate_probability, float):
            raise ValueError(f"self.mutate_probability: Debe ser un float y su tipo es {type(self.mutate_probability)}")
        if not isinstance(self.mutation_center_mean, float):
            raise ValueError(f"self.mutation_center_mean: Debe ser un float y su tipo es {type(self.mutation_center_mean)}")
        if not isinstance(self.mutation_size, float):
            raise ValueError(f"self.mutation_size: Debe ser un float y su tipo es {type(self.mutation_size)}")
        if self.mutation_size < 0:
            raise ValueError(f"self.mutation_size: Debe ser un float >= 0 y su valor es {self.mutation_size}")

        # -- Validar strings problem_type, tournament_method
        if not isinstance(self.problem_type, str):
            raise ValueError(f"self.problem_type: Debe ser un str y su tipo es {type(self.problem_type)}")
        if self.problem_type not in ["minimize", "maximize"]:
            raise ValueError(f'self.problem_type debe ser una opción de estas: ["minimize", "maximize"] y se ha pasado {self.problem_type}')
        if not isinstance(self.tournament_method, str):
            raise ValueError(f"self.tournament_method: Debe ser un str y su tipo es {type(self.tournament_method)}")
        # Valido si el tournament_method está habilitado
        if self.tournament_method not in self.GTM.get_allowed_tournament_methods():
            raise ValueError(f"self.tournament_method: El tournament_method escogido es {self.tournament_method}. "
                             f"Debe estar entre los siguientes: {self.GTM.get_allowed_tournament_methods()}")

        return True


# -- Creamos el diccionario de bounds
bounds = BoundCreator()
bounds.add_bound("learning_rate", 0.0001, 0.1, 0.000001, 1, "float")
bounds.add_bound("batch_size", 12, 64, 8, 124, "int")
print(bounds.get_bound())


print(QGO(bounds.get_bound(),
          5,
          20,
          lambda x: x + 1,
          "minimize",
          "ea_simple",
          3,
          0.2,
          0.25,
          0.0,
          0.5,
          ))
