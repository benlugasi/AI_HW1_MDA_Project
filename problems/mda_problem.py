import math
from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *
from .map_heuristics import AirDistHeuristic
from .cached_map_distance_finder import CachedMapDistanceFinder
from .mda_problem_input import *


__all__ = ['MDAState', 'MDACost', 'MDAProblem', 'MDAOptimizationObjective']


@dataclass(frozen=True)
class MDAState(GraphProblemState):
    """
    An instance of this class represents a state of MDA problem.
    This state includes:
        `current_site`:
            The current site where the ambulate is at.
            The initial state stored in this field the initial ambulance location (which is a `Junction` object).
            Other states stores the last visited reported apartment (object of type `ApartmentWithSymptomsReport`),
             or the last visited laboratory (object of type `Laboratory`).
        `tests_on_ambulance`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests are still stored on the ambulance (hasn't been transferred to a laboratory yet).
        `tests_transferred_to_lab`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests had already been transferred to a laboratory.
        `nr_matoshim_on_ambulance`:
            The number of matoshim currently stored on the ambulance.
            Whenever visiting a reported apartment, this number is decreased by the #roommates in this apartment.
            Whenever visiting a laboratory for the first time, we transfer the available matoshim from this lab
             to the ambulance.
        `visited_labs`:
            Stores the laboratories (objects of type `Laboratory`) that had been visited at least once.
    """

    current_site: Union[Junction, Laboratory, ApartmentWithSymptomsReport] #curLoc
    tests_on_ambulance: FrozenSet[ApartmentWithSymptomsReport] #Taken
    tests_transferred_to_lab: FrozenSet[ApartmentWithSymptomsReport] #Transferred
    nr_matoshim_on_ambulance: int #Matoshim
    visited_labs: FrozenSet[Laboratory] #VisitedLabs

    @property
    def current_location(self):
        if isinstance(self.current_site, ApartmentWithSymptomsReport) or isinstance(self.current_site, Laboratory):
            return self.current_site.location
        assert isinstance(self.current_site, Junction)
        return self.current_site

    def get_current_location_short_description(self) -> str:
        if isinstance(self.current_site, ApartmentWithSymptomsReport):
            return f'test @ {self.current_site.reporter_name}'
        if isinstance(self.current_site, Laboratory):
            return f'lab {self.current_site.name}'
        return 'initial-location'

    def __str__(self):
        return f'(' \
               f'loc: {self.get_current_location_short_description()} ' \
               f'tests on ambulance: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_on_ambulance]} ' \
               f'tests transferred to lab: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_transferred_to_lab]} ' \
               f'#matoshim: {self.nr_matoshim_on_ambulance} ' \
               f'visited labs: {[lab.name for lab in self.visited_labs]}' \
               f')'

    def __eq__(self, other):
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, MDAState)

        return self.current_site == other.current_site\
               and self.tests_on_ambulance == other.tests_on_ambulance\
               and self.tests_transferred_to_lab == other.tests_transferred_to_lab\
               and self.nr_matoshim_on_ambulance == other.nr_matoshim_on_ambulance\
               and self.visited_labs == other.visited_labs

    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.current_site, self.tests_on_ambulance, self.tests_transferred_to_lab,
                     self.nr_matoshim_on_ambulance, self.visited_labs))

    def get_total_nr_tests_taken_and_stored_on_ambulance(self) -> int:
        """
        This method returns the total number of of tests that are stored on the ambulance in this state.
        """
        return sum(apartment.nr_roommates for apartment in self.tests_on_ambulance)

class MDAOptimizationObjective(Enum):
    Distance = 'Distance'
    Monetary = 'Monetary'
    TestsTravelDistance = 'TestsTravelDistance'


@dataclass(frozen=True)
class MDACost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `MDAProblem.expand_state_with_costs()`.
    The `SearchNode`s that will be created during the run of the search algorithm are going
     to have instances of `MDACost` in SearchNode's `cost` field (instead of float values).
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 3 objectives:
     (i) distance, (ii) money, and (iii) tests-travel.
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 2 different costs of that solution,
     even though the objective was only one of the costs.
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
    distance_cost: float = 0.0
    monetary_cost: float = 0.0
    tests_travel_distance_cost: float = 0.0
    optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Monetary

    def __add__(self, other):
        assert isinstance(other, MDACost)
        assert other.optimization_objective == self.optimization_objective
        return MDACost(
            optimization_objective=self.optimization_objective,
            distance_cost=self.distance_cost + other.distance_cost,
            monetary_cost=self.monetary_cost + other.monetary_cost,
            tests_travel_distance_cost=self.tests_travel_distance_cost + other.tests_travel_distance_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == MDAOptimizationObjective.Distance:
            return self.distance_cost
        elif self.optimization_objective == MDAOptimizationObjective.Monetary:
            return self.monetary_cost
        assert self.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        return self.tests_travel_distance_cost

    def __repr__(self):
        return f'MDACost(' \
               f'dist={self.distance_cost:11.3f}m, ' \
               f'money={self.monetary_cost:11.3f}NIS, ' \
               f'tests-travel={self.tests_travel_distance_cost:11.3f}m)'


class MDAProblem(GraphProblem):
    """
    An instance of this class represents an MDA problem.
    """

    name = 'MDA'

    def __init__(self,
                 problem_input: MDAProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.reported_apartments)}):{optimization_objective.name})'
        initial_state = MDAState(
            current_site=problem_input.ambulance.initial_location,
            tests_on_ambulance=frozenset(),
            tests_transferred_to_lab=frozenset(),
            nr_matoshim_on_ambulance=problem_input.ambulance.initial_nr_matoshim,
            visited_labs=frozenset())
        super(MDAProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(AirDistHeuristic))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        This method represents the `Succ: S -> P(S)` function of the MDA problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The MDA problem operators are defined in the assignment instructions.
        It receives a state and iterates over its successor states.
        Notice that this its return type is an *Iterator*. It means that this function is not
         a regular function, but a `generator function`. Hence, it should be implemented using
         the `yield` statement.
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name. Look for its definition
            and use the correct fields in its c'tor. The operator name should be in the following
            format: `visit ReporterName` (with the correct reporter name) if an reported-apartment
            visit operator was applied (to take tests from the roommates of an apartment), or
            `go to lab LabName` if a laboratory visit operator was applied.
            The apartment-report object stores its reporter-name in one of its fields.
        Things you might want to use:
            - The method `self.get_total_nr_tests_taken_and_stored_on_ambulance()`.V
            - The field `self.problem_input.laboratories`.V
            - The field `self.problem_input.ambulance.total_fridges_capacity`. V
            - The method `self.get_reported_apartments_waiting_to_visit()` here. V
            - The method `self.get_operator_cost()`.V
            - The c'tor for `AmbulanceState` to create the new successor state.V
            - Python's built-in method `frozenset()` to create a new frozen set for fields thatV
              expect this type) from another collection (set/list/tuple/iterator).
            - Other fields of the state and the problem input.
            - Python's sets union operation (`some_set_or_frozenset | some_other_set_or_frozenset`).V
        """
        assert isinstance(state_to_expand, MDAState)
        sites = set(self.problem_input.laboratories) | set(self.get_reported_apartments_waiting_to_visit(state_to_expand))

        apartmentCanVisit = lambda cur_state, apt : True if(apt not in cur_state.tests_on_ambulance | cur_state.tests_transferred_to_lab\
            and apt.nr_roommates <= cur_state.nr_matoshim_on_ambulance\
            and apt.nr_roommates <= self.problem_input.ambulance.total_fridges_capacity - cur_state.get_total_nr_tests_taken_and_stored_on_ambulance())\
            else False

        labCanVisit = lambda cur_state, lab: True if(cur_state.get_total_nr_tests_taken_and_stored_on_ambulance() > 0 or lab not in cur_state.visited_labs)\
            else False

        for site in sites:
            if isinstance(site, ApartmentWithSymptomsReport):  # appt.
                if apartmentCanVisit(state_to_expand, site):
                    taken = frozenset({site}) | state_to_expand.tests_on_ambulance
                    succ_state = MDAState(site, taken, state_to_expand.tests_transferred_to_lab,
                                          state_to_expand.nr_matoshim_on_ambulance - site.nr_roommates,
                                          state_to_expand.visited_labs)
                    o_name = "visit " + site.reporter_name
                else:
                    continue
            if isinstance(site, Laboratory):
                if labCanVisit(state_to_expand, site):
                    transferred = state_to_expand.tests_on_ambulance | state_to_expand.tests_transferred_to_lab
                    nr_matoshim_in_lab = 0
                    if site not in state_to_expand.visited_labs:
                        nr_matoshim_in_lab = site.max_nr_matoshim
                    visitedLabs = state_to_expand.visited_labs | frozenset({site})
                    succ_state = MDAState(site, frozenset(), transferred,
                                          state_to_expand.nr_matoshim_on_ambulance + nr_matoshim_in_lab, visitedLabs)
                    o_name = "go to lab " + site.name
                else:
                    continue
            yield OperatorResult(succ_state, self.get_operator_cost(state_to_expand, succ_state), o_name)

    def get_operator_cost(self, prev_state: MDAState, succ_state: MDAState) -> MDACost:
        """
        Calculates the operator cost (of type `MDACost`) of an operator (moving from the `prev_state`
         to the `succ_state`). The `MDACost` type is defined above in this file (with explanations).
        Use the formal MDA problem's operator costs definition presented in the assignment-instructions.
        Use the method `self.map_distance_finder.get_map_cost_between()` to calculate the distance
         between to junctions. This distance is used for calculating the 3 costs.
        If the location of the next state is not reachable (on the streets-map) from the location of
         the previous state, use the value of `float('inf')` for all costs.
        You might want to use the method `MDAState::get_total_nr_tests_taken_and_stored_on_ambulance()`
         both for the tests-travel and the monetary costs.
        For the monetary cost you might want to use the following fields:
         `self.problem_input.ambulance.drive_gas_consumption_liter_per_meter`
         `self.problem_input.gas_liter_price`
         `self.problem_input.ambulance.fridges_gas_consumption_liter_per_meter`
         `self.problem_input.ambulance.fridge_capacity`
         `MDAState::get_total_nr_tests_taken_and_stored_on_ambulance()`
        For calculating the #active-fridges (the monetary cost) you might want to use the
         function `math.ceil(some_float_value)`.
        Note: For calculating sum of a collection (list/tuple/set) in python, you can simply
         use `sum(some_collection)`.
        Note: For getting a slice of an tuple/list in python you can use slicing indexing. examples:
            `some_tuple[:k]` - would create a new tuple with the first `k` elements of `some_tuple`.
            `some_tuple[k:]` - would create a new tuple that is based on `some_tuple` but without
                               its first `k` items.
            `some_tuple[k:n]` - would create a new tuple that is based on `some_tuple` but without
                                its first `k` items and until the `n`-th item.
            You might find this tip useful for summing a slice of a collection.
        """
        distance_cost = self.map_distance_finder.get_map_cost_between(prev_state.current_location, succ_state.current_location)
        if distance_cost is None:
            return MDACost(float('inf'), float('inf'), float('inf'), self.optimization_objective)
        testsOnAmbulance = prev_state.get_total_nr_tests_taken_and_stored_on_ambulance()
        labTestTransferCost = 0
        labRevisitCost = 0

        if isinstance(succ_state.current_site, Laboratory):
            if testsOnAmbulance > 0:
                labTestTransferCost = succ_state.current_site.tests_transfer_cost
            if succ_state.current_site in prev_state.visited_labs:
                labRevisitCost = succ_state.current_site.revisit_extra_cost

        gasPrice = self.problem_input.gas_liter_price
        driverGasCons = self.problem_input.ambulance.drive_gas_consumption_liter_per_meter
        fridgeCapacity = float(self.problem_input.ambulance.fridge_capacity)
        nrActiveFridges = math.ceil(float(testsOnAmbulance)/fridgeCapacity)
        fridgeGasCons = sum(self.problem_input.ambulance.fridges_gas_consumption_liter_per_meter[:nrActiveFridges])
        monetary_cost = (gasPrice*(driverGasCons+fridgeGasCons)*distance_cost+(labTestTransferCost+labRevisitCost))
        tests_travel_distance_cost = testsOnAmbulance*distance_cost

        return MDACost(distance_cost, monetary_cost, tests_travel_distance_cost, self.optimization_objective)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        """
        assert isinstance(state, MDAState)
        """
        Goal is:
        1. Need to be in a lab
        2. taken should be empty set // Need to visit in all apartment 
        3. Transferred = Apartments // all the apartments moved to transferred 
        4. M belongs to N // Need to have any number of Matoshim
        5. L in Labs // need to be in any Lab
                """
        return isinstance(state.current_site, Laboratory)\
               and state.tests_on_ambulance == frozenset()\
               and len(self.problem_input.reported_apartments) == len(state.tests_transferred_to_lab)

    def get_zero_cost(self) -> Cost:
        """
        Overridden method of base class `GraphProblem`. For more information, read
         documentation in the default implementation of this method there.
        In this problem the accumulated cost is not a single float scalar, but an
         extended cost, which actually includes 2 scalar costs.
        """
        return MDACost(optimization_objective=self.optimization_objective)

    def get_reported_apartments_waiting_to_visit(self, state: MDAState) -> List[ApartmentWithSymptomsReport]:
        """
        This method returns a list of all reported-apartments that haven't been visited yet.
        For the sake of determinism considerations, the returned list has to be sorted by
         the apartment's report id in an ascending order.
        """
        departments = self.problem_input.reported_apartments
        visitedApartments = set(state.tests_on_ambulance) | set(state.tests_transferred_to_lab)
        waitingForVisit = list(set(departments) - set(visitedApartments))
        waitingForVisit.sort(key=lambda x: x.report_id)
        return waitingForVisit

    def get_all_certain_junctions_in_remaining_ambulance_path(self, state: MDAState) -> List[Junction]:
        """
        This method returns a list of junctions that are part of the remaining route of the ambulance.
        This includes the ambulance's current location, and the locations of the reported apartments
         that hasn't been visited yet.
        The list should be ordered by the junctions index ascendingly (small to big).
        """
        WaitingForVisit = self.get_reported_apartments_waiting_to_visit(state)
        remaningJunctions = [state.current_location]
        for apartment in WaitingForVisit:
            remaningJunctions.append(apartment.location)
        sorted_arr = sorted(remaningJunctions, key=lambda j: j.index)
        return sorted_arr
