"""
Dynamic Augmented Adaptive Group Counting Strategies (DAPTS).

An augmented grouped test returns the exact COUNT of active individuals
in the pool, rather than just a binary nonzero_count/zero_count result.
"""

from augmented.core import (
    mask_from_indices, indices_from_mask, popcount,
    mask_str, all_pools, all_pools_from_mask,
    compute_active_mask, test_result,
)
from augmented.strategy import DAPTS, History
from augmented.simulator import apply_dapts
from augmented.expected_utility import exact_expected_utility, mc_expected_utility
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts
from augmented.bayesian import (
    bayesian_update, bayesian_update_single_test,
    bayesian_update_by_counting, gibbs_update, estimate_p_from_history,
)
from augmented.greedy import (
    greedy_myopic_simulate, greedy_myopic_expected_utility,
    greedy_lookahead_simulate,
    greedy_myopic_counting_simulate, greedy_myopic_counting_expected_utility,
    greedy_myopic_gibbs_simulate, greedy_myopic_gibbs_expected_utility,
)
from augmented.tree_extractor import extract_tree, print_tree, export_tree_dot
from augmented.static_solver import solve_static_non_overlapping, solve_static_overlapping
from augmented.classical_solver import solve_classical_dynamic
