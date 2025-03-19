from genethic_tournament_methods.qgan_reproductor import QGANReproductor
from quantum_technology import QuantumTechnology
from genethic_individuals import Individual

from typing import List


class Reproductor:

    def __init__(self, reproductor_type: str, best_individuals: List[Individual], optimizer_executor: QuantumTechnology):

        self.reproductor_type: str = reproductor_type
        self.best_individuals: List[Individual] = best_individuals
        self.optimizer_executor: QuantumTechnology = optimizer_executor

    def run(self):

        match self.reproductor_type:

            case "QGAN":
                best_children: dict = QGANReproductor(self.best_individuals, self.optimizer_executor).run_optimization_pipeline(num_samples=50, top_n=10, discriminator_epochs=300, verbose=1)
                print(best_children)