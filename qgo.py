from typing import Callable, Literal
from genethic_tournament_methods import GenethicTournamentMethods, EaSimple, EaSimpleTournament
from bounds_creator import BoundCreator
from genethic_individuals import *
from quantum_technology import QuantumTechnology

# from qiskit import QuantumCircuit, Aer, execute


class QGO:
    def __init__(self,
                 bounds_dict: Dict[str, Tuple[Union[int, float]]],
                 num_generations: int,
                 num_individuals: int,
                 objective_function: Callable,
                 tournament_method: GenethicTournamentMethods,
                 problem_type: str = "minimize",
                 podium_size: int = 3,
                 reproduction_variability: float = 0.2,
                 mutate_probability: float = 0.25,
                 mutation_center_mean: float = 0.0,
                 mutation_size: float = 0.5,
                 randomness_quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 randomness_technology: Literal["aer", "ibm"] = "aer",
                 optimizer_quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 optimizer_technology: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy",
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
        :param randomness_quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro technology. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param randomness_technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la lógica.
        :param optimizer_quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro technology. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param optimizer_technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        """

        # -- Almaceno propiedades
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.num_generations: int = num_generations
        self.num_individuals: int = num_individuals
        self.objective_function: Callable = objective_function
        self.problem_type: str = problem_type
        self.tournament_method: GenethicTournamentMethods = tournament_method
        self.podium_size: int = podium_size
        self.reproduction_variability: float = reproduction_variability
        self.mutate_probability: float = mutate_probability
        self.mutation_center_mean: float = mutation_center_mean
        self.mutation_size: float = mutation_size
        self.randomness_quantum_technology: Literal["simulator", "quantum_machine"] = randomness_quantum_technology
        self.randomness_technology: Literal["aer", "ibm"] = randomness_technology
        self.qm_api_key: str = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.optimizer_quantum_technology: Literal["simulator", "quantum_machine"] = optimizer_quantum_technology
        self.optimizer_technology: Literal["aer", "ibm"] = optimizer_technology
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine

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

        # -- Creamos los individuos y los almacenamos en una lista
        self.individuals_list: List[Individual] = []
        for i in range(self.num_individuals):
            self.individuals_list.append(Individual(self.randomness_executor, self.bounds_dict, None, 14))

        for idx, i in enumerate(self.individuals_list):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        # -- Evaluamos los resultados de primera generacion
        for individual in self.individuals_list:
            print(self.objective_function(individual))
            individual.individual_values["objective_function_values"] = self.objective_function(individual)

        for idx, i in enumerate(self.individuals_list):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        # -- Seleccionar los padres
        self.best_individuals: List[Individual] = self.tournament_method.run(self.individuals_list)
        for idx, i in enumerate(self.best_individuals):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        # -- Obtenemos los hijos a partir de los padres
        

        # -- Armar bucle de generaciones


        """# -- Entramos a la parte genetica

        # -- Se supone que ya hemos obtenido los hijos
        child_list: List[List] = [
            [0.08910494983403751, 36.5],
            [0.07170969052131343, 71.5],
            [0.005021695515233724, 16.5],
            [0.015734475416848345, 856.5],
        ]
        self.individuals_list = [Individual(self.randomness_executor, bounds_dict, child_vals) for child_vals in child_list]

        for i in self.individuals_list:
            print(f"Malformation: {i.malformation} - Values: {i.get_individual_values()}")

        # self.individuals_list = Individuals(self.bounds_dict, self.num_individuals, False, child_list).get_individuals()

        print(self.individuals_list)"""

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

        return True

    @staticmethod
    def quantum_technology(quantum_technology: str = "simulator", service: str = "aer", qm_api_key: str | None = None,
                           qm_connection_service: str | None = None, quantum_machine: str = "least_busy"):

        """
        Metodo para generar los objetos de conexión (ordenador cuántico o simulador).
        :param quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la que calculan los valores aleatorios. Si es simulator, se hará con un simulador
         definido en el parámetro randomness_technology. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro randomness_technology.
        :param service. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la selección aleatoria de variables.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        :return: QuantumTechnology
        """

        return QuantumTechnology(quantum_technology, service, qm_api_key, qm_connection_service, quantum_machine)

    def generate_individuals(self, qm_conn_object, child_values: dict | None = None, max_qubits: int = 14):
        individuals_list: List[Individual] = []
        for i in range(self.num_individuals):
            individuals_list.append(Individual(qm_conn_object, self.bounds_dict, child_values, max_qubits))
        return individuals_list

    @staticmethod
    def define_tournament():
        pass

    @staticmethod
    def mutate_tournament():
        pass

# -- Creamos el diccionario de bounds
bounds = BoundCreator()
bounds.add_bound("n_estimators", 100, 200, 50, 250, "int")
bounds.add_bound("max_depth", 2, 6, 1, 7, "int")
print(bounds.get_bound())


def objetive_function(individual: Individual):
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Cargar el dataset de diabetes
    data = load_diabetes()
    individual_dict: dict = individual.get_individual_values()
    X, y = data.data, data.target

    # Convertir la variable objetivo en un problema de clasificación binaria (diabetes alta o baja)
    y = (y > np.median(y)).astype(int)  # 1 si es mayor a la mediana, 0 si es menor

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Función objetivo para entrenar el modelo y calcular la precisión
    def train_and_evaluate_model(X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=individual_dict["n_estimators"], max_depth=individual_dict["max_depth"], random_state=42)  # Modelo Random Forest
        model.fit(X_train, y_train)  # Entrenar
        y_pred = model.predict(X_test)  # Predecir
        accuracy = accuracy_score(y_test, y_pred)  # Calcular precisión
        return accuracy

    # Entrenar y evaluar el modelo
    accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    return accuracy

ea_simple: EaSimpleTournament = EaSimpleTournament()
tournament: GenethicTournamentMethods = GenethicTournamentMethods(ea_simple)

qgo = QGO(bounds.get_bound(),
          5,
          20,
          objetive_function,
          tournament,
          "minimize",
          3,
          0.2,
          0.25,
          0.0,
          0.5,
          "simulator",
          "aer",
          "simulator",
          "aer",
          "246f573b5c03238493997c82561bf5b4e1e949b6a54f7cc3099012018e798aaf82040be8b32c0d7954363c9a5b0908dbbb9b490dfcb0d081c00915fa913b871b",
          "ibm_quantum",
          "least_busy"
          )


