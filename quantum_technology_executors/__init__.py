# -- TODO: objeto de conexion a máquinas reales y logica de simulador
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from typing import Literal, cast, List
from qiskit.compiler import transpile
import sys


class QuantumSimulator:

    def __init__(self, service: Literal["aer", "ibm"] = "aer"):
        """
        :param service. Literal["aer"]. El servicio tecnológico con el cual se ejecuta la logica.
        """

        self.service: Literal["aer", "ibm"] = service
        self.sampler = None

        if self.service == "ibm":
            sys.exit(f"Se ha seleccionado el servicio {self.service} para ejecutar el simulador cuántico (usar: aer)")

        match self.service:

            case "aer":
                self.sampler = Sampler(AerSimulator())

    def run(self, qcs: List[QuantumCircuit], shots: int):
        """
        Metodo para ejecutar un simulador cuantico y obtener sus resultados
        :param qc: Circuito cuantico que se quiere medir
        :param shots: Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # -- Definimos lista de resultados (homologado con metodo run de QuantumMachine)
        results: list = []

        # -- Definimos el sampler para ejecutar shots cantidad de veces el circuito cuantico especificado
        for qc in qcs:

            job = self.sampler.run([qc], shots=shots)

            # -- Lanzamos el job (tarea de ejecución del circuito cuántico) y obtenemos sus resultados
            job_result = job.result()[0].data.c
            results.append(job_result)

        results = self.get_results(results)

        return results

    @staticmethod
    def get_results(results: list):

        results_list: list = []

        # -- Accedemos a los valores de las mediciones del circuito cuantico
        for result in results:

            # -- Contamos la probabilidad de los resultados
            qc_ibm_results = result.get_counts()

            results_list.append(qc_ibm_results)

        return results_list


class QuantumMachine:

    def __init__(self,
                 service: Literal["aer", "ibm"],
                 qm_api_key: str | None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"],
                 optimization_level: int = 1):
        """
        :param service. ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. str | None. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Literal["ibm_quantum", "ibm_cloud"] | None. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        :param optimization_level. Nivel de optimización del circuito cuántico
        """

        self.service: Literal["aer", "ibm"] = service
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine
        self.optimization_level: int = optimization_level

        # -- Definimos la variable con la máquina elegida (puede ser igual a quantum_machine pero también la que resulte de least_busy)
        self.selected_machine: str | None = None
        self.ibm_machine_transpiler = None

        # -- Definimos el sampler de la máquina cuántica seleccionada
        self.sampler = None

        if self.service == "aer":
            sys.exit(f"Se ha seleccionado el servicio {self.service} para ejecutar el ordenador cuantico (usar: ibm)")

        match self.service:

            case "ibm":

                # -- Obtenemos el nombre de la máquina elegida y el transpilador de esa máquina
                # self.selected_machine, self.connection_transpiler = self.connection_service()
                if self.qm_connection_service == "ibm_quantum":

                    print(f"---> Conectando con el servicio de computacion cuantica de IBM: ibm_quantum")

                    # -- Generamos el servicio de conexion
                    channel_type = Literal["ibm_cloud", "ibm_quantum", "local"]
                    channel = cast(channel_type, self.qm_connection_service)
                    self.service: QiskitRuntimeService = QiskitRuntimeService(channel=channel, token=self.qm_api_key)

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

    def run(self, qc_list: List[QuantumCircuit], shots: int):

        """
        Metodo para ejecutar un ordenador cuantico y obtener sus resultados
        :param qc_list: List[QuantumCircuit]. Circuito cuantico que se quiere medir
        :param shots: int. Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # Transpilar los circuitos antes de la ejecución
        transpiled_circuits = transpile(qc_list, backend=self.selected_machine,
                                        optimization_level=self.optimization_level)

        # Usar el backend seleccionado para crear una sesión
        with Session(backend=self.selected_machine) as session:
            # Generamos el sampler de la máquina
            self.sampler = Sampler(mode=session)

            # Ejecutar el trabajo con los circuitos transpilados y el número de shots
            job = self.sampler.run(transpiled_circuits, shots=shots)

            # Esperamos a que el trabajo termine y obtener los resultados
            results = job.result()

        session.close()

        # Retornar las cuasi-distribuciones de las mediciones
        if results is not None:
           results = self.get_results(results)
           return results

        else:
            sys.exit("No se han podido obtener los resultados del ordenador cuántico. FIN")

    @staticmethod
    def get_results(results: list):
        results_list = []

        for result in results:
            counts = result.data.c.get_counts()
            results_list.append(counts)

        return results_list


    def connection_service(self, optimization_level: int = 1):
        """
        Conectamos con el servicio de computacion cuantica de IBM o trabajamos en local
        :param optimization_level: nivel de optimización del transpilador
        :return: el objeto del servicio (service) y la lista de ordenadores
        """

        if self.qm_connection_service == "ibm_quantum":

            print(f"---> Conectando con el servicio de computacion cuantica de IBM: ibm_quantum")

            # -- Generamos el servicio de conexion
            channel_type = Literal["ibm_cloud", "ibm_quantum", "local"]
            channel = cast(channel_type, self.qm_connection_service)
            self.service: QiskitRuntimeService = QiskitRuntimeService(channel=channel, token=self.qm_api_key)

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

    def __init__(self,
                 quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy",
                 ):

        """
        Metodo que instancia un objeto QuantumTechnology, el cual puede ser un simulador o un conector a una máquina cuántica real
        :param quantum_technology. Literal["simulator", "quantum_machine"]. Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro service. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param service. ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Literal["ibm_quantum", "ibm_cloud"] | None. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        """

        self.quantum_technology: Literal["simulator", "quantum_machine"] = quantum_technology
        self.service: Literal["aer", "ibm"] = service
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine

        # -- TODO: Diccionario de tecnologias y servicios habilitados (actualizar periódicamente)
        self._allowed_quantum_tech: dict = {"simulator": ["aer"],
                                            "quantum_machine": ["ibm"],
                                            "quantum_machines": {"ibm": ["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]},
                                            "quantum_services": {"ibm": ["ibm_quantum", "ibm_cloud"]}}

        # -- Validamos que los parámetros relacionados con la tecnología cuántica son correctos
        self.validate_input_parameters()

        # -- Generamos el objeto que ejecuta el algoritmo cuántico
        self.execution_object: QuantumSimulator | QuantumMachine | None = None

        match quantum_technology:

            case "simulator":
                self.execution_object: QuantumSimulator = QuantumSimulator(self.service)

            case "quantum_machine":
                self.execution_object: QuantumMachine = QuantumMachine(self.service, self.qm_api_key, self.qm_connection_service, self.quantum_machine, 1)

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
            if self.service not in self._allowed_quantum_tech["simulator"]:
                raise ValueError(f"self.quantum_technology_executor: La randomness_technology escogida es {self.quantum_technology}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['simulator']}")

        # -- Validamos el ordenador cuántico elegido
        if self.quantum_technology == "quantum_machine":

            if self.service not in self._allowed_quantum_tech["quantum_machine"]:
                raise ValueError(f"self.technology: La technology escogida es {self.service}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machine']}")

            if self.quantum_machine not in self._allowed_quantum_tech["quantum_machines"][f"{self.service}"]:
                raise ValueError(f"self.quantum_machine: La quantum_machine escogida es {self.quantum_machine}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machines'][f'{self.service}']}")

            if self.qm_connection_service not in self._allowed_quantum_tech["quantum_services"][f"{self.service}"]:
                raise ValueError(f"self.qm_connection_service: El qm_connection_service escogido es {self.qm_connection_service}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_services'][f'{self.service}']}")

        return True
