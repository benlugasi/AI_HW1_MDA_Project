from framework import *
from problems import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()


    # See documentation here:
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also Google for additional examples.

    p1, = ax1.plot(weights, total_cost,color = 'b', label = 'Solution cost')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()


    p2, = ax2.plot(weights, total_nr_expanded,color = 'r', label = '#Expanded states')

    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('#Expanded states', color='r')
    ax2.tick_params('y', colors='r')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 40_000,
                                   low_heuristic_weight: float = 0.5, high_heuristic_weight: float = 0.95):

    weights = np.linspace(low_heuristic_weight,high_heuristic_weight,n, retstep=True)
    weights = weights[0]

    arr_nr_expanded_states = []
    arr_solution_g_cost = []
    arr_result_weights = []
    for w in weights:
        astar = AStar(heuristic_type, w, max_nr_states_to_expand)
        res = astar.solve_problem(problem)
        if res.is_solution_found:
            arr_nr_expanded_states.append(res.nr_expanded_states)
            arr_solution_g_cost.append(res.solution_g_cost)
            arr_result_weights.append(w)

    plot_distance_and_expanded_wrt_weight_figure(problem.name,  arr_result_weights, arr_solution_g_cost, arr_nr_expanded_states)

# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def toy_map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.10 done
    toy_map_problem = MapProblem(streets_map, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.12 done
    astar = AStar(NullHeuristic)
    res = astar.solve_problem(toy_map_problem)
    print(res)

    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and NOT an instance of the heuristic (eg: not `MyHeuristicClass()`).

    # Ex.13
    astar = AStar(AirDistHeuristic)
    res = astar.solve_problem(toy_map_problem)
    print(res)

    # Ex.15
    run_astar_for_weights_in_range(AirDistHeuristic, toy_map_problem)


# --------------------------------------------------------------------
# ---------------------------- MDA Problem ---------------------------
# --------------------------------------------------------------------

loaded_problem_inputs_by_size = {}
loaded_problems_by_size_and_opt_obj = {}

def get_mda_problem(
        problem_input_size: str = 'small',
        optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
    if (problem_input_size, optimization_objective) in loaded_problems_by_size_and_opt_obj:
        return loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)]
    assert problem_input_size in {'small', 'moderate', 'big'}
    if problem_input_size not in loaded_problem_inputs_by_size:
        loaded_problem_inputs_by_size[problem_input_size] = MDAProblemInput.load_from_file(
            f'{problem_input_size}_mda.in', streets_map)
    problem = MDAProblem(
        problem_input=loaded_problem_inputs_by_size[problem_input_size],
        streets_map=streets_map,
        optimization_objective=optimization_objective)
    loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)] = problem
    return problem


def basic_mda_problem_experiments():
    print()
    print('Solve the MDA problem (small input, only distance objective, UniformCost).')

    # Ex.18
    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)
    uc = UniformCost()
    res = uc.solve_problem(small_mda_problem_with_distance_cost)
    print(res)

def mda_problem_with_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, A*, '
          'MaxAirDist & SumAirDist & MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.22
    astar = AStar(MDAMaxAirDistHeuristic)
    res = astar.solve_problem(moderate_mda_problem_with_distance_cost)
    print(res)

    # Ex.25
    astar = AStar(MDASumAirDistHeuristic)
    res = astar.solve_problem(moderate_mda_problem_with_distance_cost)
    print(res)

    # Ex.28
    astar = AStar(MDAMSTAirDistHeuristic)
    res = astar.solve_problem(moderate_mda_problem_with_distance_cost)
    print(res)



def mda_problem_with_weighted_astar_experiments():
    print()
    print('Solve the MDA problem (small & moderate input, only distance objective, wA*).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.30
    run_astar_for_weights_in_range(MDAMSTAirDistHeuristic, small_mda_problem_with_distance_cost)

    # Ex.30
    run_astar_for_weights_in_range(MDASumAirDistHeuristic, moderate_mda_problem_with_distance_cost)

def monetary_cost_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (monetary objectives).')

    small_mda_problem_with_monetary_cost = get_mda_problem('small', MDAOptimizationObjective.Monetary)
    moderate_mda_problem_with_monetary_cost = get_mda_problem('moderate', MDAOptimizationObjective.Monetary)

    # Ex.32
    uc = UniformCost()
    res = uc.solve_problem(small_mda_problem_with_monetary_cost)
    print(res)

    # Ex.32
    res = uc.solve_problem(moderate_mda_problem_with_monetary_cost)
    print(res)

def multiple_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (moderate input, distance & tests-travel-distance objectives).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_tests_travel_dist_cost = get_mda_problem('moderate', MDAOptimizationObjective.TestsTravelDistance)

    # Ex.35
    astar = AStar(MDATestsTravelDistToNearestLabHeuristic)
    res = astar.solve_problem(moderate_mda_problem_with_tests_travel_dist_cost)
    print(res)

    # Ex.38
    astar = AStar(MDAMSTAirDistHeuristic)
    res = astar.solve_problem(moderate_mda_problem_with_distance_cost)
    optimal_distance_cost = res.solution_g_cost
    eps = 0.6
    max_distance_cost = (1+eps)*optimal_distance_cost
    astar = AStar(MDATestsTravelDistToNearestLabHeuristic, open_criterion= lambda node: node.cost.distance_cost <= max_distance_cost)
    res = astar.solve_problem(moderate_mda_problem_with_tests_travel_dist_cost)
    print(res)


def mda_problem_with_astar_epsilon_experiments():
    print()
    print('Solve the MDA problem (small input, distance objective, using A*eps, use non-acceptable '
          'heuristic as focal heuristic).')
    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Firstly solve the problem with AStar & MST heuristic for having a reference for #devs.
    astar = AStar(MDAMSTAirDistHeuristic)
    res = astar.solve_problem(small_mda_problem_with_distance_cost)
    print(res)

    def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        if not hasattr(solver, '__focal_heuristic'):
            setattr(solver, '__focal_heuristic', MDASumAirDistHeuristic(problem=problem))
        focal_heuristic = getattr(solver, '__focal_heuristic')
        return focal_heuristic.estimate(node.state)

    # Ex.43
    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    # TODO: Create an instance of `AStarEpsilon` with the `MDAMSTAirDistHeuristic`.
    #       Solve the `small_mda_problem_with_distance_cost` with it and print the results.
    #       Use focal_epsilon=0.23, and max_focal_size=40.
    #       Use within_focal_priority_function=within_focal_h_sum_priority_function. This function
    #        (defined just above) is internally using the `MDASumAirDistHeuristic`.
    astarEpsilon = AStarEpsilon(MDAMSTAirDistHeuristic, focal_epsilon=0.23, max_focal_size=40,
                                within_focal_priority_function=within_focal_h_sum_priority_function)
    res = astarEpsilon.solve_problem(small_mda_problem_with_distance_cost)
    print(res)
    exit()  # TODO: remove!


def mda_problem_anytime_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, Anytime-A*, '
          'MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.46
    anytime_astar = AnytimeAStar(MDAMSTAirDistHeuristic, max_nr_states_to_expand_per_iteration=1000)
    res = anytime_astar.solve_problem(moderate_mda_problem_with_distance_cost)
    print(res)
    # TODO: create an instance of `AnytimeAStar` once with the `MDAMSTAirDistHeuristic`, with
    #       `max_nr_states_to_expand_per_iteration` set to 1000, solve the
    #       `moderate_mda_problem_with_distance_cost` with it and print the results.
    exit()  # TODO: remove!


def run_all_experiments():
    print('Running all experiments')
    #toy_map_problem_experiments() #todo: uncomment this when assinging the HOME Work
    #basic_mda_problem_experiments()
    #mda_problem_with_astar_experiments()
    #mda_problem_with_weighted_astar_experiments()
    #monetary_cost_objectives_mda_problem_experiments()
    #multiple_objectives_mda_problem_experiments()
    #mda_problem_with_astar_epsilon_experiments()
    mda_problem_anytime_astar_experiments()


if __name__ == '__main__':
    run_all_experiments()
