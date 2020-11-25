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
        TODO [Ex.42]: Implement this method!
            1.
            Find the minimum expanding-priority value in the `open` queue.
            Calculate the maximum expanding-priority of the FOCAL, which is
             the min expanding-priority in open multiplied by (1 + eps) where
             eps is stored under `self.focal_epsilon`.
            2.
            Create the FOCAL by popping items from the `open` queue and inserting
             them into a focal list. Don't forget to satisfy the constraint of
             `self.max_focal_size` if it is set (not None).
             3.
             Notice: You might want to pop items from the `open` priority queue,
             and then choose an item out of these popped items. Don't forget:
             the other items have to be pushed back into open.
             4.
            Inspect the base class `BestFirstSearch` to retrieve the type of
             the field `open`. Then find the definition of this type and find
             the right methods to use (you might want to peek the head node, to
             pop/push nodes and to query whether the queue is empty).
            Remember that `open` is a priority-queue sorted by `f` in an ascending
             order (small to big). Popping / peeking `open` returns the node with
             the smallest `f`.
             5.
            For each node (candidate) in the created focal, calculate its priority
             by calling the function `self.within_focal_priority_function` on it.
             This function expects to get 3 values: the node, the problem and the
             solver (self). You can create an array of these priority values. Then,
             use `np.argmin()` to find the index of the item (within this array)
             with the minimal value. After having this index you could pop this
             item from the focal (at this index). This is the node that have to
             be eventually returned.
             ______
             6. HANDLE CORNER CASES:
            Don't forget to handle correctly corner-case like when the open queue
             is empty. In this case a value of `None` has to be returned.
             ______
            Note: All the nodes that were in the open queue at the beginning of this
             method should be kept in the open queue at the end of this method, except
             for the extracted (and returned) node.
             _____
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

        #todo: check if it's equal or not
        while not self.open.is_empty() and (self.max_focal_size is None or len(focal) <= self.max_focal_size):
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

        #  V.2 self.within_focal_priority_function(node)

        focal_index = np.argmin(resArr)  # argmin returns the indices of the min value

        if isinstance(focal_index, np.intc):  # meaning only one value returned
            expended_node = focal.pop(focal_index)
        else:  # --> there are multiple indices of the same minimal value in the focal list - pick one of them
            expended_node = focal.pop(focal_index[0])
        # _________________________________________________________________________________________________

        # Part 6
        #  delete the returned node from the open queue
        self.open.extract_node(expended_node)

        return expended_node
