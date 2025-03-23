from quantum_technology import QuantumTechnology, QuantumSimulator, QuantumMachine

from typing import Literal, Dict, Union, List
from qiskit import QuantumCircuit

import warnings
import math
import sys



class Generator:

    def __init__(self,
                 num_individuals: int,
                 bounds_dict: Dict,
                 max_qubits: int = 14,
                 operation: Literal["generate", "reproduct"] = "generate",
                 quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 quantum_service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy"):

        """
        Inicializa la clase Generator con los parámetros necesarios para la generación y reproducción de individuos
        utilizando tecnología cuántica.
        :param num_individuals (int): Número de individuos a generar.
        :param bounds_dict (Dict[str, Tuple[Union[int, float]]]): Diccionario con los límites de los parámetros.
        :param max_qubits (int): Número máximo de qubits a utilizar (por defecto 14).
        :param operation (Literal["generate", "reproduct"]): Tipo de operación a realizar.
        :param quantum_technology (Literal["simulator", "quantum_machine"]): Tecnología cuántica a emplear.
        :param quantum_service (Literal["aer", "ibm"]): Servicio cuántico a utilizar.
        :param qm_api_key (str | None): Clave API para acceso a servicios cuánticos.
        :param qm_connection_service (Literal["ibm_quantum", "ibm_cloud"] | None): Servicio de conexión a IBM Quantum.
        :param quantum_machine (Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]): Máquina cuántica.
        """

        # <editor-fold desc="Definicion de variables generales de la clase  ------------------------------------------">

        # -- Definimos el tipo de tecnología y el servicio cuántico a utilizar
        self.operation: Literal["generate", "reproduct"] = operation
        self.quantum_technology: Literal["simulator", "quantum_machine"] = quantum_technology
        self.quantum_service: Literal["aer", "ibm"] = quantum_service
        self.qm_api_key: str = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine

        # -- Definimos el ejecutor de operaciones en el simulador u ordenador cuántico
        self.executor: QuantumMachine | QuantumSimulator | None = None

        # -- Definimos la cantidad, el diccionario de características a generar/reproducir y los qubits a utilizar
        self.num_individuals: int = num_individuals
        self.bounds_dict: Dict = bounds_dict
        self.max_qubits: int = max_qubits

        # -- Creamos la variable que almacenará los valores de propiedad de los individuos
        self.individual_values: Dict[str, Union[int, float]] = {}

        # -- Guardamos el numero de qubits utilizado para generar el circuito de cada propiedad por cada individuo
        self._individual_prop_num_qubits: dict = {}

        # </editor-fold>

    def generate_properties(self):

        """
        Metodo que genera las propiedades cuánticas de los individuos mediante circuitos cuánticos.
        Proceso: calcula cuántos qubits se necesitan por parámetro y ajusta valores según limitaciones malformantes.

        Retorna:
        - results_dict (dict): Diccionario con los valores generados para cada individuo.
        """

        # -- Creamos la lista de circuitos cuánticos (un circuito por propiedad de cada individuo)
        _qc_list: List[QuantumCircuit] = []

        # -- Iteramos por cada individuo del total de individuos que se buscan generar
        for individual in range(self.num_individuals):

            # -- Definimos el idx de cada individuo para almacenar sus propiedades
            self._individual_prop_num_qubits[individual] = {}

            # -- Iteramos por parámetro del diccionario de propiedades
            for parameter in self.bounds_dict.keys():

                # -- Calculamos el numero de qubits necesarios para representar los bounds (si es necesario se adaptan)
                _dynamic_max_qubits: int = self._calculate_num_qubits(bounds_dict=self.bounds_dict,
                                                                    parameter=parameter,
                                                                    max_qubits=self.max_qubits)

                # -- Almacenamos la cantidad de qubits que utilizaremos para calcular el valor de la propiedad
                self._individual_prop_num_qubits[individual][parameter] = _dynamic_max_qubits

                # -- Generamos el circuito cuantico del parametro
                _temp_qc: QuantumCircuit = self._generate_qc(_dynamic_max_qubits)

                # -- Apendeamos el circuito cuántico generado a la lista de circuitos
                _qc_list.append(_temp_qc)

        # -- Ejecutamos todos los circuitos cuánticos bajo una misma sesion
        _results = self._quantum_randomness(_qc_list)

        # -- Generamos un diccionario de resultados para adjudicar los parametros a cada individuo
        results_dict: dict = {}

        # -- Inicializamos el diccionario de individuos con los nombres de los parámetros
        individuals_dict = {
            str(i): {} for i in range(self.num_individuals)  # Inicializa cada individuo con un diccionario vacío
        }

        # -- Llenamos el diccionario con los valores de _results
        for i, result in enumerate(_results):

            # -- Extraemos la clave binaria y su cantidad
            binary_value, quantity = list(result.items())[0]

            # -- Determinamos a qué individuo pertenece este resultado
            individual_index = i // (len(_results) // self.num_individuals)

            # -- Asignamos el resultado al individuo correspondiente
            individuals_dict[str(individual_index)][binary_value] = quantity

        # -- Iteramos por individuo a generar (por su idx)
        for individual in range(0, self.num_individuals):

            print("\n######################################################################################")
            print(f"Proceso de conversion de claves binarias a enteros o flotantes: Individuo {individual}")
            print("########################################################################################\n")

            # -- Creamos el diccionario de parámetros por individuo
            results_dict[individual] = {}

            # -- Iteramos por cada parametro de los individuos a generar
            for parameter in range(0, len(self.bounds_dict.keys())):

                print(f"--------------> Parametro: {parameter}")

                # -- Obtenemos el numero binario de esta propiedad de este individuo
                try:
                    binary_num: str = [z for z in individuals_dict[str(individual)].keys()][parameter]
                except IndexError:
                    binary_num: str = [z for z in individuals_dict[str(individual)].keys()][0]

                # -- Obtenemos las claves del diccionario bounds_dict
                bounds_dict_keys: list = [z for z in self.bounds_dict.keys()]

                # -- Obtenemos el valor minimo y maximo de los bounds dict para este parámetro y el tipo de dato
                min_value: int | float = self.bounds_dict[bounds_dict_keys[parameter]]["limits"][0]
                max_value: int | float = self.bounds_dict[bounds_dict_keys[parameter]]["limits"][1]
                parameter_type: str = self.bounds_dict[bounds_dict_keys[parameter]]["type"]
                parameter_name: str = bounds_dict_keys[parameter]

                # -- Obtenemos la cantidad de qubits necesarios que se utilizaron para calcular este parámetro
                num_qubits: int = self._individual_prop_num_qubits[individual][bounds_dict_keys[parameter]]

                # -- Calculamos el valor final de cada propiedad de cada individuo
                results_dict[individual][parameter_name] = self._calculate_random_values(result=binary_num,
                                                                                    min_value=min_value,
                                                                                    max_value=max_value,
                                                                                    prop_type=parameter_type,
                                                                                    num_qubits=num_qubits)

        print("########################################################################################")
        print("########################################################################################\n")

        return results_dict

    @staticmethod
    def _calculate_num_qubits(bounds_dict: Dict, parameter: str, max_qubits: int) -> int:
        """
        Calcula la cantidad de qubits necesarios para representar un parámetro dado en un espacio de búsqueda.
        :param bounds_dict: (Dict[str, Tuple[Union[int, float]]]): Dict con límites y tipo de datos de cada parámetro.
        :param parameter: (str) Nombre del parámetro a analizar.
        :param max_qubits: (int) Número máximo de qubits disponibles.

        Returns (int): Número de qubits asignados al parámetro.
        """

        # -- Obtenemos el valor máximo de la tupla a fin de calcular la cantidad de qubits necesarios
        max_value: int | float = max(bounds_dict[parameter]["limits"])

        # -- Si el tipo de datos de la tupla son enteros, se calcula la cantidad de qubits para representarlos
        if bounds_dict[parameter]["type"] == "int":

            # -- Definir un margen adicional dinámico basado en la cantidad de qubits disponibles
            _extra_range = min(2 ** max_qubits - max_value, 20)  # Máximo extra 20, sin exceder 2^qubits

            # -- Calculamos cuántos qubits necesitamos para representar el máximo valor
            _required_qubits = math.ceil(math.log2(max_value + _extra_range))

            # -- Revisamos que el numero de qubits necesarios no sea mayor al numero de qubits que queremos utilizar
            if _required_qubits > max_qubits:
                _max_value_adjusted = 2 ** max_qubits - 1
                warnings.warn(
                    f"⚠️ Se necesitan {_required_qubits} qubits para representar {max_value} -> hay {max_qubits} qubits"
                    f"⚠️ Se ajusta el numero de qubits a {max_qubits}"
                )

            # -- Obtenemos el minimo numero de qubits necesarios entre el máximo de qubits y los requeridos
            _dynamic_max_qubits = min(max_qubits, _required_qubits)

            return _dynamic_max_qubits

        # -- Si el tipo de datos de la tupla son flotantes, se calcula la cantidad de qubits para representarlos
        elif bounds_dict[parameter]["type"] == "float":

            # -- Definimos cuántos qubits dispondremos para la parte entera y cuántos para la parte fraccionaria
            total_bits: int = max_qubits

            # -- Determinamos la cantidad mínima de qubits enteros necesarios
            required_int_bits: int = math.ceil(math.log2(max_value + 1))

            # -- Reservamos al menos 3 qubits para la parte fraccionaria
            if required_int_bits > total_bits - 3:

                # -- Ajustamos max_value para que quepa en total_bits menos los qubits de la parte fraccionaria
                _int_bits: int = total_bits - 3
                _max_value_adjusted: int = 2 ** _int_bits - 1
                warnings.warn(
                    f"⚠️ Se necesitan {required_int_bits} qubits para la parte entera de {max_value} (hay {_int_bits})."
                    f"⚠️ Se ajusta el máximo numero de qubits a {_max_value_adjusted} qubits.")
            else:
                # -- No se requiere ajuste
                _int_bits = required_int_bits
                _max_value_adjusted = max_value

            # -- Usamos el resto de qubits para la parte fraccionaria
            _frac_bits: int = total_bits - _int_bits
            _dynamic_max_qubits = _int_bits + _frac_bits

            return _dynamic_max_qubits

        else:
            sys.exit("Se está intentando calcular el número de qubits a partir de un valor no numérico. FIN")

    @staticmethod
    def _generate_qc(max_qubits: int) -> QuantumCircuit:

        # -- Creamos el circuito cuántico
        qc = QuantumCircuit(max_qubits, max_qubits)

        # -- Aplicar Hadamard a todos los qubits para lograr una superposición uniforme
        qc.h(range(max_qubits))

        # -- Medimos todos los qubits
        qc.measure(range(max_qubits), range(max_qubits))

        return qc

    def _quantum_randomness(self, qcs: List[QuantumCircuit]):
        """
        Ejecuta circuitos cuánticos para generar números aleatorios utilizando un simulador u ordenador cuántico real.
        :param qcs: (List[QuantumCircuit]) Lista de circuitos cuánticos a ejecutar.

        Return: results: (List[Dict[str: int]) Lista de resultados en formato binario.
        """

        # -- Instanciamos el ejecutor del circuito
        _executor = QuantumTechnology(quantum_technology=self.quantum_technology,
                                     service=self.quantum_service,
                                     qm_api_key=self.qm_api_key,
                                     qm_connection_service=self.qm_connection_service,
                                     quantum_machine=self.quantum_machine).get_quantum_technology()

        # -- Ejecutamos una vez el circuito cuántico con el ejecutor
        result: list = _executor.run(qcs, 1)

        return result

    @staticmethod
    def _calculate_random_values(result: str, min_value: int | float, max_value: int | float, prop_type: str,
                                 num_qubits: int = 14):
        """
        Genera un número aleatorio a partir de los números binarios btenidos de una ejecución cuántica.
        El número aleatorio puede ser de tipo entero o flotante.

        Proceso: convierte la clave binaria del resultado en un valor decimal y luego lo escala para producir un número
        aleatorio dentro del rango solicitado. Se debe destacar que en caso de haberse utilizado un ordenador cuántico,
        el número creado ha sido generado desde la aleatoriedad del colapso de la función de onda al medir el qubit.
        El metodo convierte la clave binaria obtenida en el resultado cuántico en un valor decimal, luego lo normaliza
        para asegurar que esté dentro del rango especificado. Si prop_type es int, el número generado se convierte a un
        valor entero dentro del rango. Si es `"float"`, el valor decimal se redondea a una precisión de 10 decimales.

        :param result: Dict[str: int] Dict con claves binarias (resultados) y cantidad de veces que se obtuvo.
        :param min_value: (int | float) Valor mínimo del rango en el cual se generará el número aleatorio.
        :param max_value: (int | float) Valor máximo del rango en el cual se generará el número aleatorio.
        :param prop_type: (str) Tipo de valor que se generará, int para un valor entero o float para un valor decimal.
        :param num_qubits: (int) Número de qubits usados en la simulación cuántica (opcional, valor por defecto: 14).

        :return: (int | float) El número aleatorio generado dentro del rango [min_value, max_value].
        """

        print("\n--------------------------------------------------------")

        # -- Obtenemos la primera clave del diccionario (la clave binaria)
        _binary_key = result
        print(f"La clave binaria que se está convirtiendo es {_binary_key}")

        # -- Convertimos la clave binaria a decimal (_binary_key 2 signfica que estamos en base binaria) y normalizamos
        _random_decimal = int(_binary_key, 2) / (2 ** num_qubits)
        print(f"La clave decimal de la clave binaria {_binary_key} es {_random_decimal}")

        # -- Generamos el número aleatorio dentro del rango especificado
        if prop_type == "int":

            # Calculamos el valor entero dentro del rango
            random_value = min_value + _random_decimal * (max_value - min_value)
            random_value = int(round(random_value))  # Aseguramos que sea un valor entero dentro del rango

            print(f"El valor entero final generado que se asignará al parámetro es {random_value}")
            print("--------------------------------------------------------\n")

            return random_value

        elif prop_type == "float":

            # -- Calculamos el valor flotante dentro del rango, normalizamos, y redondeamos con 10 decimales
            random_value = min_value + _random_decimal * (max_value - min_value)
            random_value = round(random_value, 10)

            print(f"El valor flotante final generado que se asignará al parámetro es {random_value}")
            print("--------------------------------------------------------\n")

            return random_value

        else:
            sys.exit(f"No se ha podido convertir la clave binaria {_binary_key} a entero o flotante. FIN")
