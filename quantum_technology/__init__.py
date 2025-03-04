# -- TODO: objeto de conexion a máquinas reales y logica de simulador
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

class QuantumSimulator:

    def __init__(self, technology: str = "aer"):

        """
        :param technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la logica.
        """

        self.technology: str = technology
        self.sampler = None

        match self.technology:

            case "aer":
                self.sampler = Sampler(AerSimulator())

    def run(self, qc, shots: int):

        """
        Metodo para ejecutar un simulador cuantico y obtener sus resultados
        :param qc: Circuito cuantico que se quiere medir
        :param shots: Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # -- Definimos el sampler para ejecutar shots cantidad de veces el circuito cuantico especificado
        job = self.sampler.run([qc], shots=shots)

        # -- Lanzamos el job (tarea de ejecución del circuito cuántico) y obtenemos sus resultados
        results = job.result()

        # -- Accedemos a los valores de las mediciones del circuito cuantico
        qc_results = results[0].data.c

        # -- Contamos la probabilidad de los resultados
        qc_results = qc_results.get_counts()

        return qc_results



class QuantumMachine:

    def __init__(self, technology: str, qm_api_key: str, qm_connection_service: str, quantum_machine: str, optimization_level: int = 1):
        """
        :param technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha: ibm_quantum | ibm_cloud
        :param quantum_machine. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        :param optimization_level. Nivel de optimización del circuito cuántico
        """

        self.technology: str= technology
        self.qm_api_key: str= qm_api_key
        self.qm_connection_service: str = qm_connection_service
        self.quantum_machine: str = quantum_machine
        self.optimization_level: int = optimization_level

        # -- Definimos la variable con la máquina elegida (puede ser igual a quantum_machine pero también la que resulte de least_busy)
        self.selected_machine: str | None = None

        # -- Definimos el sampler de la máquina cuántica seleccionada
        self.sampler = None

        match self.technology:

            case "ibm":

                # -- Obtenemos el nombre de la máquina elegida y el transpilador de esa máquina
                # self.selected_machine, self.connection_transpiler = self.connection_service()
                if self.qm_connection_service == "ibm_quantum":

                    print(f"---> Conectando con el servicio de computacion cuantica de IBM: ibm_quantum")

                    # -- Generamos el servicio de conexion
                    self.service: QiskitRuntimeService = QiskitRuntimeService(channel=self.qm_connection_service, token=self.qm_api_key)

                    # -- Chequeamos si la conexion ha sido exitosa
                    if self.service.active_account()["verify"]:
                        print("---> Conexion realizada con exito")

                        print("---> Datos de la cuenta")
                        _user_data: dict = self.service.usage()
                        _user_period_start: str = _user_data["period"]["start"]
                        _user_period_end: str = _user_data["period"]["end"]
                        _by_instance: str = _user_data["byInstance"][0]["instance"]
                        _user_quota: int = _user_data["byInstance"][0]["quota"]
                        _user_usage: int = _user_data["byInstance"][0]["usage"]
                        _user_pending_jobs: int = _user_data["byInstance"][0]["pendingJobs"]
                        _user_max_pending_jobs: int = _user_data["byInstance"][0]["maxPendingJobs"]

                        print(f"---> Instancia de ejecucion: {_by_instance}")
                        print(f"---> Cuota de ejecucion: {_user_quota}")
                        print(f"---> Usos del usuario: {_user_usage}")
                        print(f"---> Trabajos pendientes: {_user_pending_jobs} / {_user_max_pending_jobs}")

                        # -- Buscamos la maquina elegida
                        print("---> Buscando ordenador cuantico...")
                        if self.quantum_machine == "least_busy":
                            print(f"---> Buscando la maquina menos cargada...")
                            least_busy_machine = self.service.least_busy()
                            self.selected_machine = self.service.backend(least_busy_machine.name)
                            print(f"---> La maquina menos cargada es: {self.selected_machine.name}")
                        else:
                            print(f"---> Buscando la maquina {self.quantum_machine}...")
                            self.selected_machine = self.service.backend(self.quantum_machine)

                        print(f"---> Numero de Qubits: {self.selected_machine.num_qubits}")
                        print(f"---> Trabajos pendientes: {self.selected_machine.status().pending_jobs}")
                        print(f"---> Operaciones permitidas: {self.selected_machine.operation_names}")
                        print(f"---> Numero maximo de circuitos: {self.selected_machine.max_circuits}")

                        if self.selected_machine is not None:

                            # -- Generamos el transpilador o pass manager de la maquina elegida
                            self.connection_transpiler = generate_preset_pass_manager(backend=self.selected_machine, optimization_level=self.optimization_level)

                        else:
                            print("El ordenador especificado no existe. FIN")
                            sys.exit()

                elif self.qm_connection_service == "ibm_cloud":
                    pass

                # -- Generamos el sampler de la máquina
                self.sampler = Sampler(self.selected_machine)

    def run(self, qc, shots: int):

        """
        Metodo para ejecutar un ordenador cuantico y obtener sus resultados
        :param qc: Circuito cuantico que se quiere medir
        :param shots: Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # -- Traspilamos el circuito cuántico a la máquina concreta de IBM elegida
        qc_circuit = self.connection_transpiler.run(qc)

        # -- Definimos el job para ejecutar el circuito cuantico segun las cantidad de shots
        job = self.sampler.run([qc_circuit], shots=shots)

        # -- Lanzamos el job (tarea de ejecución del circuito cuántico) y obtenemos sus resultados
        results = job.result()

        # -- Accedemos a los valores de las mediciones del circuito cuantico
        qc_ibm_results = results[0].data.c

        # -- Contamos la probabilidad de los resultados
        qc_ibm_results = qc_ibm_results.get_counts()

        return qc_ibm_results

    def connection_service(self, optimization_level: int = 1):
        """
        Conectamos con el servicio de computacion cuantica de IBM o trabajamos en local
        :param optimization_level: nivel de optimización del transpilador
        :return: el objeto del servicio (service) y la lista de ordenadores
        """

        if self.qm_connection_service == "ibm_quantum":

            print(f"---> Conectando con el servicio de computacion cuantica de IBM: ibm_quantum")

            # -- Generamos el servicio de conexion
            self.service: QiskitRuntimeService = QiskitRuntimeService(channel=self.qm_connection_service, token=self.qm_api_key)

            # -- Chequeamos si la conexion ha sido exitosa
            if self.service.active_account()["verify"]:
                print("---> Conexion realizada con exito")

                print("---> Datos de la cuenta")
                _user_data: dict = self.service.usage()
                _user_period_start: str = _user_data["period"]["start"]
                _user_period_end: str = _user_data["period"]["end"]
                _by_instance: str = _user_data["byInstance"][0]["instance"]
                _user_quota: int = _user_data["byInstance"][0]["quota"]
                _user_usage: int = _user_data["byInstance"][0]["usage"]
                _user_pending_jobs: int = _user_data["byInstance"][0]["pendingJobs"]
                _user_max_pending_jobs: int = _user_data["byInstance"][0]["maxPendingJobs"]

                print(f"---> Instancia de ejecucion: {_by_instance}")
                print(f"---> Cuota de ejecucion: {_user_quota}")
                print(f"---> Usos del usuario: {_user_usage}")
                print(f"---> Trabajos pendientes: {_user_pending_jobs} / {_user_max_pending_jobs}")

                # -- Buscamos la maquina elegida
                print("---> Buscando ordenador cuantico...")
                if self.quantum_machine == "least_busy":
                    print(f"---> Buscando la maquina menos cargada...")
                    least_busy_machine = self.service.least_busy()
                    self.selected_machine = self.service.backend(least_busy_machine.name)
                    print(f"---> La maquina menos cargada es: {self.selected_machine.name}")
                else:
                    print(f"---> Buscando la maquina {self.quantum_machine}...")
                    self.selected_machine = self.service.backend(self.quantum_machine)

                print(f"---> Numero de Qubits: {self.selected_machine.num_qubits}")
                print(f"---> Trabajos pendientes: {self.selected_machine.status().pending_jobs}")
                print(f"---> Operaciones permitidas: {self.selected_machine.operation_names}")
                print(f"---> Numero maximo de circuitos: {self.selected_machine.max_circuits}")

                if self.selected_machine is not None:

                    # -- Generamos el transpilador o pass manager de la maquina elegida
                    self.ibm_machine_transpiler = generate_preset_pass_manager(backend=self.selected_machine, optimization_level=optimization_level)

                else:
                    print("El ordenador especificado no existe. FIN")
                    sys.exit()

                return self.selected_machine, self.ibm_machine_transpiler

        elif self.qm_connection_service == "ibm_cloud":
            return None

class QuantumTechnology:

    def __init__(self, quantum_technology: str = "simulator", technology: str = "aer", qm_api_key: str | None = None, qm_connection_service: str | None = None,
                 quantum_machine: str = "least_busy"):

        """
        Metodo que instancia un objeto QuantumTechnology, el cual puede ser un simulador o un conector a una máquina cuántica real
        :param quantum_technology. [simulator, quantum_machine] Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro technology. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param technology. ["aer", "ibm", "d-wave", etc.] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha: ibm_quantum | ibm_cloud
        :param quantum_machine. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        """

        self.quantum_technology: str = quantum_technology
        self.technology: str = technology
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: str | None = qm_connection_service
        self.quantum_machine: str = quantum_machine

        # -- TODO: Diccionario de tecnologias y servicios habilitados (actualizar periódicamente)
        self._allowed_quantum_tech: dict = {"simulator": ["aer"],
                                            "quantum_machine":["ibm"],
                                            "quantum_machines": {"ibm": ["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]},
                                            "quantum_services": {"ibm": ["ibm_quantum", "ibm_cloud"]}}

        # -- Validamos que los parámetros relacionados con la tecnología cuántica son correctos
        self.validate_input_parameters()

        # -- Generamos el objeto que ejecuta el algoritmo cuántico
        self.execution_object: Simulator | QuantumMachineConnector | None = None

        match quantum_technology:

            case "simulator":
                self.execution_object: Simulator = QuantumSimulator(self.technology)

            case "quantum_machine":
                self.execution_object: QuantumMachine = QuantumMachine(self.technology, self.qm_api_key, self.qm_connection_service, self.quantum_machine, 1)

    def get_quantum_technology(self):
        """
        Metodo getter para retorna el execution object (objeto que ejecuta un circuito en un ordenador cuántico o en un simulador cuántico)
        :return: self.execution_object
        """
        return self.execution_object

    def validate_input_parameters(self) -> bool:
        """
        Metodo para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # -- Validamos la tecnologia cuántica definida
        if self.quantum_technology == "simulator":
            if self.technology not in self._allowed_quantum_tech["simulator"]:
                raise ValueError(f"self.quantum_technology: La randomness_technology escogida es {self.quantum_technology}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['simulator']}")

        # -- Validamos el ordenador cuántico elegido
        if self.quantum_technology == "quantum_machine":

            if self.technology not in self._allowed_quantum_tech["quantum_machine"]:
                raise ValueError(f"self.technology: La technology escogida es {self.technology}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machine']}")

            if self.quantum_machine not in self._allowed_quantum_tech["quantum_machines"][f"{self.technology}"]:
                raise ValueError(f"self.quantum_machine: La quantum_machine escogida es {self.quantum_machine}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machines'][f'{self.technology}']}")

            if self.qm_connection_service not in self._allowed_quantum_tech["quantum_services"][f"{self.technology}"]:
                raise ValueError(f"self.qm_connection_service: El qm_connection_service escogido es {self.qm_connection_service}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_services'][f'{self.technology}']}")

        return True

    def quantum_random_real(self, min_value: int | float, max_value: int | float, num_qubits: int =14):
        """
        Genera un número aleatorio cuántico entre min_value y max_value.

        Parámetros:
        - min_value: Límite mínimo del rango
        - max_value: Límite máximo del rango
        - num_qubits: Número de qubits para la generación (por defecto 14)

        Retorna:
        Un número aleatorio entre min_value y max_value
        """

        # -- Creamos el circuito cuántico
        qc = QuantumCircuit(num_qubits, num_qubits)

        # -- Aplicar Hadamard a todos los qubits para lograr una superposición uniforme
        qc.h(range(num_qubits))

        # -- Medimos todos los qubits
        qc.measure(range(num_qubits), range(num_qubits))

        # -- Ejecutamos el circuito
        result = self.execution_object.run(qc, 1)

        # -- Obtenemos los resultados
        result = list(result.keys())[0]

        # -- Convertimos el numero binario a decimal y lo normalizamo entre [0,1]
        random_decimal = int(result, 2) / (2 ** num_qubits)

        # -- Obtenemos el numero cuántico aleaotorio buscado
        return min_value + random_decimal * (max_value - min_value)