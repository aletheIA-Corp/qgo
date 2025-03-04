from genethic_tournament_methods.ea_simple import EaSimple
from genethic_tournament_methods.ea_generate_update import EaGenerateUpdate
from genethic_tournament_methods.ea_mu_comma_lambda import EaMuCommaLambda
from genethic_tournament_methods.ea_mu_plus_lambda import EaMuPlusLambda
from typing import List


class GenethicTournamentMethods:
    def __init__(self):
        self._allowed_tournament_methods: List[str] = ["ea_simple", "ea_generate_update", "ea_mu_comma_lambda", "ea_mu_plus_lambda"]

    def get_allowed_tournament_methods(self) -> List[str]:
        return self._allowed_tournament_methods


