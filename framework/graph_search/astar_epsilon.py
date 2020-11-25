from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math


class AStarEpsilon(AStar):
    """
    This class implements the (weighted) A*Epsilon search algorithm.
    A*Epsilon algorithm basically works like the A* algorithm, but with
    another way to choose the next node to expand from the open queue.
    """

    solver_name = 'A*eps'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 within_focal_priority_function: Callable[[SearchNode, GraphProblem, 'AStarEpsilon'], float],
                 heuristic_weight: float = 0.5,
                 max_nr_states_to_expand: Optional[int] = None,
                 focal_epsilon: float = 0.1,
                 max_focal_size: Optional[int] = None):
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStarEpsilon, self).__init__(heuristic_function_type, heuristic_weight,
                                           max_nr_states_to_expand=max_nr_states_to_expand)
        self.focal_epsilon = focal_epsilon
        if focal_epsilon < 0:
            raise ValueError(f'The argument `focal_epsilon` for A*eps should be >= 0; '
                             f'given focal_epsilon={focal_epsilon}.')
        self.within_focal_priority_function = within_focal_priority_function
        self.max_focal_size = max_focal_size

    def _init_solver(self, problem):
        super(AStarEpsilon, self)._init_solver(problem)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         by focusing on the current FOCAL and choosing the node
         with the best within_focal_priority from it.
        """
        if self.open.is_empty():  # first Corner Case
            return None

        # 1._______________________________
        min_open_val = self.open.peek_next_node().expanding_priority  # based on 4, the min is at the beginning
        max_expanding_priority = min_open_val * (1 + self.focal_epsilon)

        # 2._______________________________
        # inits
        focal = []
        restorer_open_list = []

        while not self.open.is_empty() and (self.max_focal_size is None or len(focal) < self.max_focal_size):
            node_to_expand = self.open.pop_next_node()
            if node_to_expand.expanding_priority <= max_expanding_priority:
                focal.append(node_to_expand)  # says in the comment that focal is a list
            restorer_open_list.append(node_to_expand)

        # 3. + 4. --> adjustments to restore open
        while restorer_open_list:
            self.open.push_node(restorer_open_list.pop(0))
        # ___________________________________
        # 5. ___________________________ Working on the Focal list ________________________________________________

        if not focal:
            return None

        resArr = np.array([])  # init empty numpy arr

        for node in focal:
            resArr = np.append(resArr, float(self.within_focal_priority_function(node, problem, self)))

        focal_index = np.atleast_1d(np.argmin(resArr))
        expended_node = focal.pop(focal_index[0])
        # _________________________________________________________________________________________________

        # Part 6
        #  delete the returned node from the open queue
        self.open.extract_node(expended_node)

        if self.use_close:
            self.close.add_node(expended_node)

        return expended_node
