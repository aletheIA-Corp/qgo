from genetic_tournament import GenethicTournamentMethods, EaSimpleTournament
from genetic_individuals import Individual, Population
from bounds_creator import BoundCreator
from mutation_methods import Mutation

from typing import Callable, Literal, List, Tuple, Dict, Union

import warnings


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
                 reproductor: Literal["QGAN"] | None = None,
                 max_attempts_fill_population: int = 3,
                 verbose: bool = True):

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

        max_attempts_fill_population : int, opcional
            Cantidad de intentos en los que se interará rellenar la población de la primera generación

        Verbose: bool
            Verbose para imprimir información adicional en consola
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
        self.reproductor: Literal["QGAN"] | None = reproductor
        self.max_attempts_fill_population: int = max_attempts_fill_population
        self.current_gen: int = 0
        self.verbose = verbose

        # -- Validamos los inputs del constructor
        self.validate_input_parameters()

        # </editor-fold>

        # <editor-fold desc="Creamos la primera generacion de individuos  --------------------------------------------">
        print("\n################################## INICIO ###############################################")
        print(f"1. Creamos la primera generación de individuos de la población")
        print("################################## INICIO ###############################################\n")

        # -- Inicializamos la variable en la cual guardaremos/administraremos todos los individuos de cada generación
        self.population: Population = Population()

        # -- Inicializamos la variable en la que guardaremos/administraremos los individuos ganadores de cada generación
        self.winner_population: Population = Population()

        # -- Definimos la primera generación
        self.generation = 0

        # -- Poblamos la población de la generación 0
        self.population.populate(generation=self.generation,
                                 operation="generate",
                                 num_individuals=self.num_individuals,
                                 bounds_dict=self.bounds_dict,
                                 max_qubits=self.max_qubit_random_generation,
                                 quantum_technology=self.randomness_quantum_technology ,
                                 quantum_service=self.randomness_service,
                                 qm_api_key=self.qm_api_key,
                                 qm_connection_service=qm_connection_service,
                                 quantum_machine=self.quantum_machine,
                                 verbose=self.verbose)

        # -- Imprimimos los individuos que ya forman parte de la poblacion de la primera generacion
        self.population.print_population(generation=self.generation)

        # -- ----------------------------------------------------------------------------------------------------------
        # -- Si se crearon individuos iguales intentamos crear otros individuos nuevos para completar la poblacion
        # -- ----------------------------------------------------------------------------------------------------------

        # -- Definimos el contador de intentos para completar la población de la primera generación de indiviudos
        attempts = 0

        # -- Si la cantidad de indiviudos de la primera generación es menor al número de indiviudos esperados, y
        # -- Si el numero de intentos es menor al número de intentos predefinidos en el constructor de QGO...
        while (len(self.population.get_individuals(self.generation)) < self.num_individuals and
               attempts < self.max_attempts_fill_population):

            # -- Obtenemos el numero de individuos que faltan para completar la poblacion de la primera generación
            num_new_individuals: int = self.num_individuals - len(self.population.get_individuals(self.generation))

            print("\n" + "#" * 90)
            print(f"Intento {attempts + 1} para generar individuos restantes (faltan {num_new_individuals} individuos)")
            print("#" * 90 + "\n")

            # -- Generamos nuevos individuos y en caso de cumplir con los estándares, los agregamos a la población
            self.population.populate(generation=self.generation,
                                     operation="generate",
                                     num_individuals=num_new_individuals,
                                     bounds_dict=self.bounds_dict,
                                     max_qubits=self.max_qubit_random_generation,
                                     quantum_technology=self.randomness_quantum_technology,
                                     quantum_service=self.randomness_service,
                                     qm_api_key=self.qm_api_key,
                                     qm_connection_service=qm_connection_service,
                                     verbose=self.verbose)

            # -- Incrementamos el contador de intentos
            attempts += 1

            self.population.print_population(generation=self.generation)

        # -- ----------------------------------------------------------------------------------------------------------
        # -- Si luego de intentar crear los individuos faltantes no se pudo completar la poblacion de la generación...
        # -- ----------------------------------------------------------------------------------------------------------

        if len(self.population.get_individuals(self.generation)) < self.num_individuals:
            warnings.warn(f"No se han podido generar todos los individuos deseados ({num_individuals}).")
            warnings.warn(f"La generación posee {len(self.population.get_individuals(self.generation))} individuos.")
            warnings.warn(f"Recomendaciones: Aumentar el rango de las propiedades/malformaciones en el bounds_dict.")
            warnings.warn(f"Recomendaciones: Disminuir el numero de invidiuos a crear para el mismo bounds_dict.")

        print("\n################################## FIN ###############################################")
        print(f"1. Creamos la primera generación de individuos de la población")
        print("################################## FIN ###############################################\n")

        # </editor-fold>

        # <editor-fold desc="Ejecutamos la función objetivo para cada individuo  -------------------------------------">
        print("\n################################## INICIO ###############################################")
        print(f"2. Ejecutamos la función objetivo para cada uno de los individuos creados de la primera generación")
        print("################################## INICIO ###############################################\n")

        # -- Obtenemos los resultados de la función de coste de la primera generacion
        for individual in self.population.get_individuals(self.generation):
            individual.add_or_update_variable("objective_function_values", self.objective_function(individual))

        self.population.print_population(generation=self.generation)

        print("\n################################## FIN ###############################################")
        print(f"2. Ejecutamos la función objetivo para cada uno de los individuos creados de la primera generación")
        print("################################## FIN ###############################################\n")

        # </editor-fold>

        # <editor-fold desc="Selección de padres, generacion de hijos y nuevos valores de función objetivo  ----------">
        for gen in range(1, self.num_generations):

            print("\n################################## INICIO ###############################################")
            print("#########################################################################################")
            print(f"3. Empezamos la generación {gen}: seleccionamos padres, generamos hijos y evaluamos nuevamente")
            print("#########################################################################################")
            print("################################## INICIO ###############################################\n")

            # <editor-fold desc="Seleccionamos los mejores padres  ---------------------------------------------------">
            print("\n################################## INICIO ###############################################")
            print(f"3.1. Seleccionamos los mejores padres utilizando el criterio del torneo instanciado")
            print("################################## INICIO ###############################################\n")

            # -- Seleccionar los padres
            self.best_individuals: List[Individual] = self.tournament_method.run(self.population.get_individuals(gen-1))

            if str(gen - 1) not in self.winner_population.get_individuals():
                self.winner_population.get_individuals()[str(gen - 1)] = []  # Inicializa la lista si no existe

            # -- Agregamos los padres ganadores a winner_population
            for individual in self.best_individuals:
                self.winner_population.get_individuals(gen-1).append(individual)

            self.winner_population.print_population(generation=gen-1)

            print("\n################################## FIN ###############################################")
            print(f"3.1. Seleccionamos los mejores padres utilizando el criterio del torneo instanciado")
            print("################################## FIN ###############################################\n")

            # </editor-fold>

            # <editor-fold desc="Generacion de hijos  ----------------------------------------------------------------">

            print("\n################################## INICIO ###############################################")
            print(f"3.2. Generamos {self.num_individuals} hijos a partir del reproductor {self.reproductor}")
            print("################################## INICIO ###############################################\n")

            # -- 1. Generamos los individuos de la generación 1 a partir de los padres de la generación 0

            # -- Incluimos los nuevos individuos en la población
            self.population.populate(generation=gen,
                                     operation="reproduct",
                                     num_individuals=self.num_individuals,
                                     bounds_dict=self.bounds_dict,
                                     max_qubits=self.max_qubit_random_generation,
                                     quantum_technology=self.optimizer_quantum_technology,
                                     quantum_service=self.optimizer_service,
                                     qm_api_key=self.qm_api_key,
                                     qm_connection_service=qm_connection_service,
                                     individuals_to_reproduct=self.winner_population.get_individuals(gen-1),
                                     reproductor=self.reproductor,
                                     verbose=self.verbose)

            self.population.print_population(gen)


            # -- 2. Ejecutamos las mutaciones genéticas de los individuo

            # -- 2.1. Mutan forzosamente los individuos repetidos
            Mutation(self.bounds_dict,
                     self.population.get_individuals(gen),
                     self.mutate_probability).mutate_repeated_individuals(verbose=True)

            # -- 2.2. Seleccionamos aleatoriamente cuántos individuos, cuáles y en qué proporción mutan sus genes
            Mutation(self.bounds_dict,
                     self.population.get_individuals(gen),
                     self.mutate_probability).run_mutation()

            # -- 2.3. Mutamos forzosamente nuevamente a los que se han repetido al mutar a todos
            Mutation(self.bounds_dict,
                     self.population.get_individuals(gen),
                     self.mutate_probability).mutate_repeated_individuals(verbose=True)

            print("\n################################## FIN ###############################################")
            print(f"3.2. Generamos {self.num_individuals} hijos a partir del reproductor {self.reproductor}")
            print("################################## FIN ###############################################\n")

            # </editor-fold>

            # <editor-fold desc="Ejecutamos la función objetivo con los hijos  ---------------------------------------">

            print("\n################################## INICIO ###############################################")
            print(f"3.3. Ejecutamos la función objetivo para cada uno de los individuos creados")
            print("################################## INICIO ###############################################\n")

            # -- Obtenemos los resultados de la función de coste de la primera generacion
            for individual in self.population.get_individuals(gen):
                individual.add_or_update_variable("objective_function_values", self.objective_function(individual))

            self.population.print_population(generation=gen)

            print("\n################################## FIN ###############################################")
            print(f"3.3. Ejecutamos la función objetivo para cada uno de los individuos creados")
            print("################################## FIN ###############################################\n")

            # </editor-fold>

        # </editor-fold>

    def validate_input_parameters(self) -> bool:
        """
        Metodo para validar los inputs que se han cargado en el constructor.
        :return: True si todas las validaciones son correctas, de lo contrario lanza una excepción.
        """

        # -- Validar bounds_dict
        if not isinstance(self.bounds_dict, dict):
            raise ValueError("bounds_dict: Debe ser un diccionario.")

        for param, param_data in self.bounds_dict.items():
            if not isinstance(param, str):
                raise ValueError(f"bounds_dict: Las claves deben ser str, pero se encontró {type(param)}.")
            if not isinstance(param_data, dict):
                raise ValueError(
                    f"bounds_dict: Los valores deben ser diccionarios, pero se encontró {type(param_data)} para {param}.")
            for key in ["limits", "malformation_limits"]:
                if key in param_data:
                    if not isinstance(param_data[key], tuple):
                        raise ValueError(f"bounds_dict[{param}]: '{key}' debe ser una tupla.")
                    if not all(isinstance(valor, (int, float)) for valor in param_data[key]):
                        raise ValueError(f"bounds_dict[{param}]: '{key}' debe contener solo int o float.")

        # -- Validar num_generations, num_individuals, podium_size, max_qubit_random_generation (Enteros)
        for param in ["num_generations", "num_individuals", "podium_size", "max_qubit_random_generation"]:
            value = getattr(self, param)
            if not isinstance(value, int):
                raise ValueError(f"{param}: Debe ser un entero y su tipo es {type(value)}")
            if value <= 0:
                raise ValueError(f"{param}: Debe ser un entero positivo y su valor es {value}")

        # -- Validar floats: reproduction_variability, mutate_probability, mutation_center_mean, mutation_size
        for param in ["reproduction_variability", "mutate_probability", "mutation_center_mean", "mutation_size"]:
            value = getattr(self, param)
            if not isinstance(value, float):
                raise ValueError(f"{param}: Debe ser un float y su tipo es {type(value)}")
        if self.mutation_size < 0:
            raise ValueError(f"mutation_size: Debe ser un float >= 0 y su valor es {self.mutation_size}")

        # -- Validar problem_type
        if not isinstance(self.problem_type, str):
            raise ValueError(f"problem_type: Debe ser un str y su tipo es {type(self.problem_type)}")
        if self.problem_type not in ["minimize", "maximize"]:
            raise ValueError(f'problem_type debe ser "minimize" o "maximize", pero se pasó {self.problem_type}')

        # -- Validar tournament_method
        if not isinstance(self.tournament_method, GenethicTournamentMethods):
            raise ValueError(
                f"tournament_method: Debe ser una instancia de GenethicTournamentMethods, pero se recibió {type(self.tournament_method)}")

        # -- Validar objective_function
        if not callable(self.objective_function):
            raise ValueError("objective_function: Debe ser una función callable.")

        # -- Validar randomness_quantum_technology y optimizer_quantum_technology
        for param in ["randomness_quantum_technology", "optimizer_quantum_technology"]:
            value = getattr(self, param)
            if value not in ["simulator", "quantum_machine"]:
                raise ValueError(f'{param}: Debe ser "simulator" o "quantum_machine", pero se pasó {value}')

        # -- Validar randomness_service y optimizer_service
        for param in ["randomness_service", "optimizer_service"]:
            value = getattr(self, param)
            if value not in ["aer", "ibm"]:
                raise ValueError(f'{param}: Debe ser "aer" o "ibm", pero se pasó {value}')

        # -- Validar qm_api_key (Puede ser None o un string)
        if self.qm_api_key is not None and not isinstance(self.qm_api_key, str):
            raise ValueError(f"qm_api_key: Debe ser None o un str, pero se recibió {type(self.qm_api_key)}")

        # -- Validar qm_connection_service (Puede ser None o un valor válido)
        if self.qm_connection_service is not None and self.qm_connection_service not in ["ibm_quantum", "ibm_cloud"]:
            raise ValueError(
                f'qm_connection_service: Debe ser None, "ibm_quantum" o "ibm_cloud", pero se pasó {self.qm_connection_service}')

        # -- Validar quantum_machine
        if self.quantum_machine not in ["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]:
            raise ValueError(
                f'quantum_machine: Debe ser uno de ["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"], pero se pasó {self.quantum_machine}')

        # -- Validar reproductor
        if self.reproductor != "QGAN":
            raise ValueError(f'reproductor: Solo se acepta "QGAN", pero se pasó {self.reproductor}')

        return True

# -- Creamos el diccionario de bounds
bounds = BoundCreator()
bounds.add_interval_bound("n_estimators", 100, 1000, 50, 1500, "int")
bounds.add_predefined_bound("max_depth", (1, 2, 3, 4, 5, 6, 7, 8, 9), "int")

print("\n################################## INICIO ###############################################")
print(f"Bounds definidos para el problema de optimización")
print("################################## INICIO ###############################################\n")

for bound in bounds.get_bound():
    print(f"{bound}: {bounds.get_bound()[bound]}")

print("\n################################## FIN ###############################################")
print(f"Bounds definidos para el problema de optimización")
print("################################## FIN ###############################################\n")

# -- Definimos la función objetivo
def objetive_function(individual: Individual) -> float:

    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # -- Cargamos el dataset de diabetes
    data = load_diabetes()
    individual_dict: dict = individual.get_individual_values()
    X, y = data.data, data.target

    # -- Convertimos la variable objetivo en un problema de clasificación binaria (diabetes alta o baja)
    y = (y > np.median(y)).astype(int)  # 1 si es mayor a la mediana, 0 si es menor

    # -- Dividimos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -- Normalizamos los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Función objetivo para entrenar el modelo y calcular la precisión
    def train_and_evaluate_model(individual_dict, X_train_scaled, X_test_scaled, y_train, y_test):

        model = RandomForestClassifier(n_estimators=individual_dict["n_estimators"],
                                       max_depth=individual_dict["max_depth"],
                                       random_state=42)

        # -- Entrenamos el modelo, predecimos y calculamos el accuracy
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    # -- Entrenamos y evaluamos el modelo
    accuracy = train_and_evaluate_model(individual_dict, X_train_scaled, X_test_scaled, y_train, y_test)

    print(f"Entrenamiento del modelo con estas propiedades: {', '.join(f'{k}: {v}' for k, v in individual_dict.items())}: Accuracy: {accuracy}")
    return accuracy

print("\n################################## INICIO ###############################################")
print(f"Instanciamos el tipo de torneo que regirá la elección de los mejores padres de cada generación")
print("################################## INICIO ###############################################\n")

# -- Definimos el tipo de torneo en el que competirán los individuos y lo instanciamos
ea_simple: EaSimpleTournament = EaSimpleTournament()
tournament: GenethicTournamentMethods = GenethicTournamentMethods(ea_simple)

print("\n################################## INICIO ###############################################")
print(f"Instanciamos el optimizado genético cuántico")
print("################################## INICIO ###############################################\n")

# -- Inicializamos el quantum genetic optimizer
qgo = QGO(bounds.get_bound(),
          8,
          50,
          objetive_function,
          tournament,
          "minimize",
          3,
          0.2,
          0.15,
          0.0,
          0.5,
          "simulator",  # -- quantum_machine | simulator
          "aer",  # -- ibm | aer
          40,
          "simulator",
          "aer",
          "246f573b5c03238493997c82561bf5b4e1e949b6a54f7cc3099012018e798aaf82040be8b32c0d7954363c9a5b0908dbbb9b490dfcb0d081c00915fa913b871b",
          "ibm_quantum",
          "least_busy",
          "QGAN",
          3,
          True
          )


