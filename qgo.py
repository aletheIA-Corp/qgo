from typing import Callable, Literal
from genethic_tournament_methods import GenethicTournamentMethods, EaSimple, EaSimpleTournament
from bounds_creator import BoundCreator
from genethic_individuals import *
from genethic_tournament_methods.reproductor import Reproductor
from quantum_technology import QuantumTechnology
from quantum_methods import Generator

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
                 randomness_service: Literal["aer", "ibm"] = "aer",
                 max_qubit_random_generation: int = 40,
                 optimizer_quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 optimizer_service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy",
                 reproductor: Literal["QGAN"] = "QGAN"
                 ):

        """
        Clase base para implementar un Algoritmo Genético Cuántico (QGA), basado en QAOA y generación de aleatoriedad cuántica
        para la evolución de la población a través de mutaciones y cruces.

        Parámetros:
        ----------
        bounds_dict : Dict[str, Tuple[Union[int, float]]]
            Diccionario que define los parámetros a optimizar y sus respectivos rangos de valores.
            Ejemplo: {'learning_rate': (0.0001, 0.1)}

        num_generations : int
            Número total de generaciones a ejecutar en el algoritmo.

        num_individuals : int
            Número de individuos en la población inicial.

        objective_function : Callable
            Función objetivo utilizada para evaluar y puntuar a cada individuo. Debe retornar un valor `float`.

        problem_type : str, opcional
            Tipo de optimización a realizar. Puede ser 'minimize' o 'maximize'.
            Ejemplo: minimizar para MAE, maximizar para Accuracy.

        tournament_method : GenethicTournamentMethods
            Metodo de selección utilizado para elegir los individuos que se reproducirán.

        podium_size : int, opcional
            Número de individuos que compiten en cada torneo para seleccionar al mejor.
            Por ejemplo, si es 3, se escogen 3 individuos al azar y se selecciona al mejor en cada iteración.

        reproduction_variability : float, opcional
            También conocido como α ∈ [0,1]. Controla la variabilidad genética entre padres e hijos.
            - Si α=0, los hijos solo pueden tomar valores interpolados entre los genes de los padres.
            - Si α>0, se permite una mayor exploración, permitiendo genes fuera del rango de los padres.

        mutate_probability : float, opcional
            Probabilidad de mutación para cada gen (`indpb` ∈ [0,1]).
            - Un valor de 0 significa que no hay mutaciones.
            - Un valor de 1 implica que siempre hay mutaciones.

        mutation_center_mean : float, opcional
            Desplazamiento medio (μ) aplicado a la mutación.
            - Si μ=0, hay igual probabilidad de mutación positiva y negativa.
            - Si μ>0, aumenta la probabilidad de mutación positiva.
            - Si μ<0, aumenta la probabilidad de mutación negativa.

        mutation_size : float, opcional
            Desviación estándar (σ) de la mutación. Controla la magnitud de los cambios en los genes.
            - 0.1 - 0.5: Mutaciones pequeñas y estables.
            - 0.5 - 1.0: Balance entre estabilidad y exploración.
            - 1.5 - 3.0: Exploración agresiva.

        randomness_quantum_technology : Literal["simulator", "quantum_machine"], opcional
            Tecnología utilizada para generar números aleatorios.
            - 'simulator': Utiliza un simulador clásico.
            - 'quantum_machine': Usa hardware cuántico real.

        randomness_service : Literal["aer", "ibm"], opcional
            Servicio de computación cuántica utilizado para la generación de aleatoriedad.

        max_qubit_random_generation : int, opcional
            Número máximo de qubits permitidos para la generación de números aleatorios.

        optimizer_quantum_technology : Literal["simulator", "quantum_machine"], opcional
            Tecnología cuántica utilizada para la optimización (simulador o máquina cuántica real).

        optimizer_service : Literal["aer", "ibm"], opcional
            Proveedor de servicios para la computación cuántica en el proceso de optimización.

        qm_api_key : str | None, opcional
            Clave de API para acceder a servicios de computación cuántica.

        qm_connection_service : Literal["ibm_quantum", "ibm_cloud"] | None, opcional
            Plataforma de IBM utilizada para la conexión cuántica.

        quantum_machine : Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"], opcional
            Máquina cuántica específica a utilizar. Si se elige 'least_busy', se seleccionará la menos ocupada.

        reproductor : Literal["QGAN"], opcional
            Metodo de reproducción utilizado en el algoritmo. Actualmente solo se admite "QGAN".
        """

        # <editor-fold desc="Definicion de variables generales de la clase  ------------------------------------------">

        # -- Almacenamos las propiedades generales de la clase
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
        self.randomness_service: Literal["aer", "ibm"] = randomness_service
        self.max_qubit_random_generation: int = max_qubit_random_generation
        self.qm_api_key: str = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.optimizer_quantum_technology: Literal["simulator", "quantum_machine"] = optimizer_quantum_technology
        self.optimizer_service: Literal["aer", "ibm"] = optimizer_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine
        self.reproductor: str = reproductor

        # -- Validamos los inputs del constructor
        self.validate_input_parameters()

        # -- Generamos el diccionario en el que almacenaremos las propiedades de cada indiduo
        self.population_properties: dict = {}

        # </editor-fold>

        # <editor-fold desc="Creamos la primera generacion de individuos  --------------------------------------------">

        # -- Generamos las propiedades de los individuos
        self.population_properties = Generator(operation="generate",
                                               num_individuals=self.num_individuals,
                                               bounds_dict=self.bounds_dict,
                                               max_qubits=self.max_qubit_random_generation,
                                               quantum_technology=self.randomness_quantum_technology ,
                                               quantum_service=self.randomness_service,
                                               qm_api_key=self.qm_api_key,
                                               qm_connection_service=qm_connection_service).generate_properties()


        # -- Inicializamos la lista vacía donde guardaremos los individuos válidos de la poblacion
        self.population: List[Individual] = []

        # -- Iteramos sobre las propiedades de los individuos
        for properties in self.population_properties.values():

            # -- Creamos individuos con las propiedades calculadas con los circuitos cuánticos
            new_individual = Individual(bounds_dict=self.bounds_dict, properties=properties, generation=0)

            # Verificamos si el individuo tiene algún valor de malformación
            individual_values = new_individual.get_individual_values()
            if not individual_values.get("malformation"):
                # Comprobamos si ya existe un individuo similar en la lista
                is_duplicate = False
                for existing_individual in self.population:
                    if existing_individual == new_individual:  # Comparación de igualdad
                        is_duplicate = True
                        break

                # Si no es un duplicado, lo agregamos a la lista
                if not is_duplicate:
                    self.population.append(new_individual)

        # -- Imprimir los individuos finales
        for idx, i in enumerate(self.population):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        # -- Generamos individuos adicionales hasta completar la cantidad deseada
        while len(self.population) < self.num_individuals:
            self.population.append(self.generate_valid_individual())

        # -- Imprimir los individuos finales
        for idx, i in enumerate(self.population):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        breakpoint()
        # </editor-fold>

        # -- Evaluamos los resultados de primera generacion
        for individual in self.population:
            print(self.objective_function(individual))
            individual.get_individual_values()["objective_function_values"] = self.objective_function(individual)

        for idx, i in enumerate(self.population):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        # -- Seleccionar los padres
        # -- TODO: nos quedamos con los mejores? Con cuántos?
        self.best_individuals: List[Individual] = self.tournament_method.run(self.population)
        for idx, i in enumerate(self.best_individuals):
            print(f"individuo_{idx}: {i.get_individual_values()}")

        # -- Obtenemos los hijos a partir de los padres
        # -- TODO: Creamos el reproductor
        reproductor = Reproductor( self.reproductor, self.best_individuals, self.optimizer_executor).run()
        print(reproductor)
        breakpoint()
        children: List[Individual] = reproductor.get_children()

        # -- Armar bucle de generaciones"""


        """# -- Entramos a la parte genetica

        # -- Se supone que ya hemos obtenido los hijos
        child_list: List[List] = [
            [0.08910494983403751, 36.5],
            [0.07170969052131343, 71.5],
            [0.005021695515233724, 16.5],
            [0.015734475416848345, 856.5],
        ]
        self.population = [Individual(self.randomness_executor, bounds_dict, child_vals) for child_vals in child_list]

        for i in self.population:
            print(f"Malformation: {i.malformation} - Values: {i.get_individual_values()}")

        # self.population = Individuals(self.bounds_dict, self.num_individuals, False, child_list).get_individuals()

        print(self.population)"""

    def validate_input_parameters(self) -> bool:
        """
        Metodo para validar los inputs que se han cargado en el constructor
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

    def generate_valid_individual(self):
        """Genera y retorna un individuo válido sin malformación."""
        while True:
            new_props = Generator(
                operation="generate",
                num_individuals=1,
                bounds_dict=self.bounds_dict,
                max_qubits=self.max_qubit_random_generation,
                quantum_technology=self.randomness_quantum_technology,
                quantum_service=self.randomness_service,
                qm_api_key=self.qm_api_key,
                qm_connection_service=self.qm_connection_service
            ).generate_properties()

            new_individual = Individual(bounds_dict=self.bounds_dict, properties=list(new_props.values())[0], generation=0)

            if not new_individual.get_individual_values().get("malformation"):
                return new_individual

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

print(f"Los bounds definidos son: {bounds.get_bound()}")

def objetive_function(individual: Individual):
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Cargar el dataset de diabetes
    data = load_diabetes()
    individual_dict: dict = individual.get_individual_values()
    X, y = data.data, data.target

    # Convertir la variable objetivo en un problema de clasificación binaria (diabetes alta o baja)
    y = (y > np.median(y)).astype(int)  # 1 si es mayor a la mediana, 0 si es menor

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Función objetivo para entrenar el modelo y calcular la precisión
    def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
        model = RandomForestClassifier(n_estimators=individual_dict["n_estimators"], max_depth=individual_dict["max_depth"], random_state=42)  # Modelo Random Forest
        model.fit(X_train_scaled, y_train)  # Entrenar
        y_pred = model.predict(X_test_scaled)  # Predecir
        accuracy = accuracy_score(y_test, y_pred)  # Calcular precisión
        return accuracy

    # Entrenar y evaluar el modelo
    accuracy = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)

    return accuracy

ea_simple: EaSimpleTournament = EaSimpleTournament()
tournament: GenethicTournamentMethods = GenethicTournamentMethods(ea_simple)

qgo = QGO(bounds.get_bound(),
          5,
          10,
          objetive_function,
          tournament,
          "minimize",
          3,
          0.2,
          0.25,
          0.0,
          0.5,
          "simulator",  # -- quantum_machine
          "aer",  # -- ibm
          40,
          "simulator",
          "aer",
          "246f573b5c03238493997c82561bf5b4e1e949b6a54f7cc3099012018e798aaf82040be8b32c0d7954363c9a5b0908dbbb9b490dfcb0d081c00915fa913b871b",
          "ibm_quantum",
          "least_busy",
          "QGAN"
          )


