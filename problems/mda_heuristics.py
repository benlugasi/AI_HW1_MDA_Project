import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator

__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDATestsTravelDistToNearestLabHeuristic']


class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Max-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the distance of the remaining path of the ambulance,
         by calculating the maximum distance within the group of air distances between each two
         junctions in the remaining ambulance path. We don't consider laboratories here because we
         do not know what laboratories would be visited in an optimal solution.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        max_air_dist = max(self.cached_air_distance_calculator.get_air_distance_between_junctions(j1, j2)
                           for j1 in all_certain_junctions_in_remaining_ambulance_path
                           for j2 in all_certain_junctions_in_remaining_ambulance_path
                           if j1.index != j2.index)
        return max_air_dist


class MDASumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining ambulance route in the following way:
        It builds a path that starts in the current ambulance's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all certain junctions (in `all_certain_junctions_in_remaining_ambulance_path`) that haven't
         been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like enforcing the #matoshim and free
         space in the ambulance's fridge). We only make sure to visit all certain junctions in
         `all_certain_junctions_in_remaining_ambulance_path`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)

        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        cur_min_pair = (0, state.current_location.index)
        path_sum = 0
        #self.problem.streets_map[cur_min_pair[1]] // should return the junction object

        while len(all_certain_junctions_in_remaining_ambulance_path) > 1: # to have at least two junctions to run on
            cur_junction = self.problem.streets_map[cur_min_pair[1]]
            all_certain_junctions_in_remaining_ambulance_path.remove(cur_junction) # this junction always in result list
            all_dist_pairs = []
            # calculate all distances pairs between the current min junction and the all the rest appt.
            for j in all_certain_junctions_in_remaining_ambulance_path:
                cur_dist = (self.cached_air_distance_calculator.get_air_distance_between_junctions(cur_junction, j), j.index)
                all_dist_pairs.append(cur_dist)
            cur_min_pair = min(all_dist_pairs)
            path_sum += cur_min_pair[0]

        return path_sum


class MDAMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MST-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the distance of the remaining route of the ambulance.
        Here this remaining distance is bounded (from below) by the weight of the minimum-spanning-tree
         of the graph, in-which the vertices are the junctions in the remaining ambulance route, and the
         edges weights (edge between each junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        G = nx.Graph()
        G.add_nodes_from(junctions)
        #for j in junctions:
        #    G.add_node(j)
        junctions_met = []
        for j1 in junctions:
            junctions_met.append(j1)
            for j2 in junctions:
                if j2 not in junctions_met:
                    G.add_edge(j1, j2, weight = self.cached_air_distance_calculator.get_air_distance_between_junctions(j1, j2))
        T = nx.minimum_spanning_tree(G)
        return T.size(weight = 'weight')

class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound to the remained tests-travel-distance of the remained ambulance path.
        The main observation is that driving from a laboratory to a reported-apartment does not increase the
         tests-travel-distance cost. So the best case (lowest cost) is when we go to the closest laboratory right
         after visiting any reported-apartment.
        If the ambulance currently stores tests, this total remained cost includes the #tests_on_ambulance times
         the distance from the current ambulance location to the closest lab.
        The rest part of the total remained cost includes the distance between each non-visited reported-apartment
         and the closest lab (to this apartment) times the roommates in this apartment (as we take tests for all
         roommates).
        TODO [Ex.33]:
            Complete the implementation of this method.
            Use `self.problem.get_reported_apartments_waiting_to_visit(state)`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        def air_dist_to_closest_lab(junction: Junction) -> float:
            """
            Returns the distance between `junction` and the laboratory that is closest to `junction`.
            """
            return min(self.cached_air_distance_calculator.get_air_distance_between_junctions(junction, lab.location)
                       for lab in self.problem.problem_input.laboratories)

        path_sum = 0
        reported_apartments_waiting_to_visit = self.problem.get_reported_apartments_waiting_to_visit(state)
        nrTestsOnAmbulance = state.get_total_nr_tests_taken_and_stored_on_ambulance()
        if nrTestsOnAmbulance > 0:
            path_sum += nrTestsOnAmbulance * air_dist_to_closest_lab(state.current_location)
        for apartment in reported_apartments_waiting_to_visit:
            dist = air_dist_to_closest_lab(apartment.location)
            path_sum += apartment.nr_roommates*dist
        return path_sum
