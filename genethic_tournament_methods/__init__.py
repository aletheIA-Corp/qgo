import random

from genethic_individuals import Individual
from genethic_tournament_methods.ea_simple import EaSimple
from genethic_tournament_methods.ea_generate_update import EaGenerateUpdate
from genethic_tournament_methods.ea_mu_comma_lambda import EaMuCommaLambda
from genethic_tournament_methods.ea_mu_plus_lambda import EaMuPlusLambda
from typing import List


class GenethicTournamentMethods:
    def __init__(self, tournament):
        self._allowed_tournament_methods: List[str] = ["ea_simple", "ea_generate_update", "ea_mu_comma_lambda", "ea_mu_plus_lambda"]
        self.tournament = tournament

    def run(self, individuals_list: List[Individual]):
        return self.tournament.run(individuals_list)

    def get_allowed_tournament_methods(self) -> List[str]:
        return self._allowed_tournament_methods


class EaSimpleTournament:

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def run(self, individuals_list: List[Individual]):
        winners: List[Individual] = []
        working_list: List[Individual] = individuals_list.copy()  # Trabajamos con una copia para no modificar la original

        while len(working_list) >= self.tournament_size:
            # -- Seleccionar 'tournament_size' elementos al azar (en este caso 3)
            selected_indices = random.sample(range(len(working_list)), self.tournament_size)
            selected_elements = [working_list[i] for i in selected_indices]

            # -- Encontrar el elemento con mayor 'objective_function_values'
            winner = max(selected_elements, key=lambda x: x.get_individual_values()["objective_function_values"])
            winners.append(winner)

            # -- Eliminar los elementos seleccionados de la lista original
            for index in sorted(selected_indices, reverse=True):  # Orden descendente para evitar desajustes
                working_list.pop(index)

        # -- Agregar cualquier elemento restante directamente a los ganadores
        winners.extend(working_list)

        return winners