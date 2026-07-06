# %% [markdown]
# ## Requirements

# %%
import pandas as pd
import numpy as np
import random
import math
import itertools
import ast
import time
from math import prod
from scipy.stats import bernoulli
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import product
from itertools import chain
from collections import defaultdict
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
import torch
from joblib import Parallel, delayed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import ast
from mosek.fusion import *
import math
from scipy.optimize import brentq
from math import log, exp
import gurobipy as gp
from gurobipy import GRB
from collections import Counter
import multiprocessing

# Total number of CPUs available on the system
print("Total CPUs available:", os.cpu_count())

# CPUs available in the SLURM allocation
print("CPUs allocated by SLURM:", multiprocessing.cpu_count())

n_jobs = multiprocessing.cpu_count()
print(f"n_jobs: {n_jobs}")

plt.rcParams['figure.dpi'] = 300


B = 2
k = 2
N = 3
G = N


def create_agents(N = N, uIntegers=False):
  agents = list()

  for i in range(N):
    if uIntegers:
      agents.append((i, round(random.random() * 100) + 1, random.random()))
    else:
      agents.append((i, random.random(), random.random()))
  return agents

agents = create_agents()


def generate_binary_tree(strategy):
    # Base case: if the strategy is empty, return None
    if not strategy:
        return None

    # Extract the current test and its nonzero_count and zero_count branches
    current_test = set([person[0] for person in strategy[0]])
    nonzero_count_branch = generate_binary_tree(strategy[1])
    zero_count_branch = generate_binary_tree(strategy[2])

    # Return a tuple representing the binary tree node
    return (current_test, nonzero_count_branch, zero_count_branch)

# %%
def tree_to_boolean_list(tree, N, bool_list=None):

    if bool_list is None:
        bool_list = []

    new_list = [False] * N

    # Unpack the tree node
    values, left, right = tree

    # Update the boolean list for this level
    for val in values:
        new_list[val] = True

    bool_list += new_list

    # Recurse for the left and right subtrees, moving to the next level
    if left:
        bool_list = tree_to_boolean_list(left, N, bool_list)
    if right:
        bool_list = tree_to_boolean_list(right, N, bool_list)

    return bool_list


def boolean_list_to_tree(bool_list, N):

    # Extract the current node values
    current_values = {i for i in range(N) if bool_list[i]}

    if len(bool_list) == N:
        return (current_values, None, None)

    # Move the index forward by N for the next segment
    cutoff = N + ((len(bool_list) - N) // 2)

    leftList = bool_list[N : cutoff]
    rightList = bool_list[cutoff:]

    # Recursively construct the left subtree
    left_subtree = boolean_list_to_tree(leftList, N)

    # Recursively construct the right subtree
    right_subtree = boolean_list_to_tree(rightList, N)

    return (current_values, left_subtree, right_subtree)


def maxUtil(agents):
  util = 0
  for agent in agents:
    util += agent[1] * agent[2]
  return util

def solveStaticNoPool(agents, B=B):
  utils = []
  for agent in agents:
    utils.append(agent[1] * agent[2])
  utils = sorted(utils, reverse=True)
    # Take the first B elements and sum them
  util = sum(utils[:B])
  return util


def solveStaticNonOverlap(agents, G = G, B = B):

  def generate_non_overlapping_subsets(agents, G, B):
    # Generate all possible subsets of agents with sizes up to G
    subsets = []
    for r in range(1, G+1):
        subsets.extend(combinations(agents, r))

    # Generate combinations of B non-overlapping subsets
    combos = combinations(subsets, B)

    # Filter combinations to keep only non-overlapping ones
    non_overlapping_combinations = []
    for combination in combos:
        combined_set = set()
        is_non_overlapping = True
        for subset in combination:
            if combined_set.intersection(subset):
                is_non_overlapping = False
                break
            combined_set.update(subset)
        if is_non_overlapping:
            non_overlapping_combinations.append(combination)

    return non_overlapping_combinations

  def strategyUtilitySNO(combination):
    utility = 0
    for group in combination:
      groupUtility = 0
      groupClearancey = 1
      for person in group:
        groupUtility += person[1]
        groupClearancey *= person[2]
      utility += groupClearancey * groupUtility
    return utility

  strategy = dict()
  combos = generate_non_overlapping_subsets(agents, G, B)
  for combination in combos:
    strategy[combination] = strategyUtilitySNO(combination)
  optimal = max(strategy, key=strategy.get)
  return optimal, strategy[optimal]

def bayesTheorem(agents, posGroups, negAgents):

  posGroups = list(posGroups)
  posGroups = sorted(posGroups, key=len)

  for posGroup in posGroups:
    for posGroupCompare in posGroups:
      if posGroup.issubset(posGroupCompare) and posGroup != posGroupCompare:
        posGroups.remove(posGroupCompare)

  newAgents = []

  for agent in agents:
    if agent[0] in set(negAgents):
      newAgents.append((agent[0], 0, 1))
    else:
      newAgents.append(agent)

  agentDict = {}
  for agent in newAgents:
    agentDict[agent[0]] = (agent[1], agent[2])


  def all_combinations(frozen_sets):

    def pick_at_least_one(agents_set):
      # Generate combinations of different sizes from the set
      combinations_list = []
      for r in range(1, len(agents_set) + 1):
          combinations_list.extend(combinations(agents_set, r))
      return combinations_list

    possibilities = [pick_at_least_one(group) for group in frozen_sets]
    product_result = product(*possibilities)

    # Combine the elements of each tuple into a single set
    combos = {frozenset(chain(*comb)) for comb in product_result}

    return combos

  def getProb(groups):
    involvedAgents = frozenset(chain(*groups))

    possiblePos = all_combinations(groups)

    prob = 0
    for possibility in possiblePos:
      currentOutcome = 1
      for id in involvedAgents:
        if id in possibility:
          currentOutcome *= 1 - agentDict[id][1]
        else:
          currentOutcome *= agentDict[id][1]
      prob += currentOutcome

    return prob

  probPos = {}

  probPos[tuple(posGroups)] = getProb(posGroups)
  probPos[()] = 1

  finalAgents = []

  for agent in newAgents:

    # P(agent is Pos | posGroups) = P(posGroups | agent is Pos) * P(agent in Pos) / P(posGroups)
    # = P(pos Groups agent not in) * P(agent is Pos) / P(posGroups)

    if agent[2] == 1 or agent[2] == 0:
      finalAgents.append(agent)
      # agentDict[agent[0]] = (agent[1], agent[2])

    else:

      posNotIn = [posGroup for posGroup in posGroups if agent[0] not in posGroup]

      if tuple(posNotIn) not in probPos:

        probPos[tuple(posNotIn)] = getProb(posNotIn)

      pAgentPos = probPos[tuple(posNotIn)] * (1 - agent[2]) / probPos[tuple(posGroups)]
      clearance = 1 - pAgentPos

      finalAgents.append((agent[0], agent[1], clearance))
      # agentDict[agent[0]] = (agent[1], clearance)

  for agent in finalAgents:
    agentDict[agent[0]] = (agent[1], agent[2])

  return agentDict


"""
MICOP model for a single test that can be solved exactly with MOSEK.

Note that the objective value of this model is the log of the overall welfare!
"""
def solveConicSingle(agents, G=G, verbose=False):

    agents = [agent for agent in agents if agent[1] != 0 and agent[2] != 0]

    if len(agents) == 0:
        return tuple(), 0

    u = [agent[1] for agent in agents]
    q = [agent[2] for agent in agents]

    # Compute population size
    n = len(u)

    assert n == len(q), "Input vectors have different lengths."
    assert all(u[i] >= 0 for i in range(n)), "Utilities must be non-zero_count."
    assert all(q[i] >= 0 and q[i] <= 1 for i in range(n)), "Probabilities must lie between 0 and 1."

    # Hack alert: Conic program doesn't like -math.inf
    q = [(math.log(qi) if qi!= 0 else -100000000000) for qi in q]

    with Model('ceo1') as M:
        # Define variables
        x = M.variable("x", n, Domain.binary())
        y = M.variable("y", 1, Domain.unbounded())
        z = M.variable("z", 1, Domain.unbounded())
        d = M.variable("d", 1, Domain.equalsTo(1))
        t = Var.vstack(z.index(0), d.index(0), y.index(0))

        # Add objective
        M.objective("obj", ObjectiveSense.Maximize, Expr.add(y.index(0), Expr.dot(x, q)))

        # Add constraints
        M.constraint("ev", Expr.sub(Expr.dot(u, x), z.index(0)), Domain.equalsTo(0))
        M.constraint("expc", t, Domain.inPExpCone())

        # Adaptive group counting size constraint

        M.constraint("pool", Expr.sum(x), Domain.lessThan(G))

        M.solve()

        # print(agents)
        # print(tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1))
        # print(M.primalObjValue())

        utility = math.exp(M.primalObjValue())
        strategy = tuple(agent for agent, level in zip(agents, np.round_(x.level())) if level == 1)

        return strategy, utility

def GibbsMCMCWindowCount(
    agents,
    posGroups,          # list of (set_of_agent_ids, exact_count)
    negAgents,          # set of forced-clearancey agents
    max_iterations=1000,
    tolerance=0.05,
    min_burn_in=50,
    window_size=100,
    confidence_level=0.95,
    n_bootstrap=1000,
    random_seed=None
):
    """
    Single-block enumerative generator for EXACT constraints, with 
    a rolling-window check for convergence similar to your original Gibbs approach.
    
    1) Identify all unknown agents (those in any posGroup).
    2) Enumerate all valid global assignments of those unknown agents that satisfy EXACT constraints,
       weighting each by product-of-priors.
    3) On each iteration, draw one assignment from that enumerated distribution. 
       Store it in a rolling window for each agent.
       After min_burn_in, check if the fraction of clearancey (0) in the last window_size is stable.
       If stable for all agents, conclude burn-in.
    4) Then do additional draws for final probabilities + confidence intervals.

    Returns: 
       agentDict[agent_id] = (utility, p_clearancey, (ci_lower, ci_upper))
    """
    import numpy as np
    from itertools import product
    from scipy.stats import bernoulli

    if random_seed is not None:
        np.random.seed(random_seed)

    # ------------------------------------------------------------
    # 1) Identify unknown agents and build quick lookups
    # ------------------------------------------------------------
    all_unknown_ids = set()
    for (grp, cval) in posGroups:
        all_unknown_ids.update(grp)
    all_unknown_ids = sorted(all_unknown_ids)

    # agent_map for quick access
    agent_map = {a[0]: a for a in agents}  # agent_id -> (id, utility, prior_active)

    # We'll store current "best guess" states for forced-clearancey or non-unknown agents,
    # but for unknowns, we draw new assignments each iteration.
    clearance_states = {}
    for (a_id, util, p_active) in agents:
        if a_id in negAgents:
            clearance_states[a_id] = 0
        else:
            # Just do a random initial state from prior (though it's not crucial, 
            # because each iteration we draw a full assignment for unknown agents).
            clearance_states[a_id] = bernoulli.rvs(p_active)

    # ------------------------------------------------------------
    # 2) Enumerate all valid assignments for the unknown agents
    #    that satisfy EXACT constraints.
    #    Each assignment is a dict: agent_id -> 0/1
    # ------------------------------------------------------------
    def is_valid(assignment_dict):
        """Check if assignment_dict satisfies forced clearancey + EXACT constraints."""
        for (grp, exact_count) in posGroups:
            # forced-clearancey must remain 0
            for forced_id in negAgents:
                if forced_id in grp and assignment_dict[forced_id] == 1:
                    return False
            # Check exact_count
            active_in_group = sum(assignment_dict[a] for a in grp)
            if active_in_group != exact_count:
                return False
        return True

    def assignment_weight(assignment_dict):
        """Product of priors for each agent's assigned state."""
        w = 1.0
        for a_id, state in assignment_dict.items():
            if a_id in negAgents and state == 1:
                return 0.0  # forced-clearancey conflict
            # agent's prior_active
            _, _, p_inf = agent_map[a_id]
            if state == 1:
                w *= p_inf
            else:
                w *= (1 - p_inf)
        return w

    # Build a baseline dict for all agents, using the current clearance_states
    # Then for each combination over unknown IDs, override them.
    # (In practice, we only need to vary the unknown ones.)
    valid_assignments = []
    for bits in product([0, 1], repeat=len(all_unknown_ids)):
        proposal = dict(clearance_states)  # copy current states
        # Overwrite unknown IDs with bits
        conflict = False
        for i, a_id in enumerate(all_unknown_ids):
            if a_id in negAgents and bits[i] == 1:
                conflict = True
                break
            proposal[a_id] = bits[i]
        if conflict:
            continue

        if is_valid(proposal):
            valid_assignments.append(proposal)

    if not valid_assignments:
        # No valid assignment => contradictory constraints
        agentDict = {}
        for (a_id, util, p_inf) in agents:
            if a_id in negAgents:
                agentDict[a_id] = (0, 1.0, (1.0, 1.0))
            else:
                agentDict[a_id] = (util, 1 - p_inf, (1 - p_inf, 1 - p_inf))
        return agentDict

    # Precompute weights
    assignment_list = []
    weights = []
    for a in valid_assignments:
        w = assignment_weight(a)
        assignment_list.append(a)
        weights.append(w)

    weights = np.array(weights, dtype=float)
    if weights.sum() == 0.0:
        # fallback to uniform if product-of-priors is zero everywhere
        weights = np.ones_like(weights)
    weights /= weights.sum()

    def draw_assignment():
        idx = np.random.choice(len(assignment_list), p=weights)
        return assignment_list[idx]

    # ------------------------------------------------------------
    # 3) Rolling window structures
    # ------------------------------------------------------------
    # We'll do up to max_iterations for burn-in, each iteration drawing
    # a new assignment => store each agent's state in a rolling buffer.
    clearance_history = {a_id: [] for a_id in all_unknown_ids}

    # ------------------------------------------------------------
    # 4) Burn-in with rolling window convergence check
    # ------------------------------------------------------------
    converged_iteration = max_iterations  # if we never break
    for iteration in range(max_iterations):
        # Draw a global assignment from the valid distribution
        chosen = draw_assignment()
        # Update our clearance_states with that assignment
        # (In principle, the forced-clearancey or unaffected agents remain as they are, but let's keep them consistent.)
        for a_id in chosen:
            clearance_states[a_id] = chosen[a_id]

        # Update rolling windows
        for a_id in all_unknown_ids:
            h = clearance_history[a_id]
            h.append(clearance_states[a_id])
            if len(h) > window_size:
                h.pop(0)

        # Check convergence if iteration > min_burn_in
        if iteration > min_burn_in:
            # We'll see if all unknown agents are stable
            stable = True
            for a_id in all_unknown_ids:
                h = clearance_history[a_id]
                if len(h) >= window_size:
                    avg_full = np.mean(h)
                    avg_half = np.mean(h[-(window_size//2):])
                    if abs(avg_full - avg_half) > tolerance:
                        stable = False
                        break
            if stable:
                converged_iteration = iteration
                break

    # That means `converged_iteration` is effectively our "burn_in".
    burn_in = converged_iteration

    # ------------------------------------------------------------
    # 5) Post-Burn-in drawing
    # ------------------------------------------------------------
    # We'll collect at least the same number of draws as burn_in or (max_iterations - burn_in)
    # like your original code
    n_post_draws = max(max_iterations - burn_in, burn_in)

    # We'll track for each agent how many times it's clearancey
    cleared_counts = {a_id: 0 for a_id in all_unknown_ids}
    post_assignments = {a_id: [] for a_id in all_unknown_ids}

    for _ in range(n_post_draws):
        chosen = draw_assignment()
        for a_id in chosen:
            clearance_states[a_id] = chosen[a_id]

        # Record each unknown agent's state
        for a_id in all_unknown_ids:
            if clearance_states[a_id] == 0:
                cleared_counts[a_id] += 1
            post_assignments[a_id].append(clearance_states[a_id])

    # Probability(clearancey) for unknown agents
    final_probs = {
        a_id: cleared_counts[a_id] / n_post_draws
        for a_id in all_unknown_ids
    }

    # ------------------------------------------------------------
    # 6) Bootstrap confidence intervals
    # ------------------------------------------------------------
    ci_intervals = {}
    for (a_id, util, p_inf) in agents:
        if a_id in negAgents:
            ci_intervals[a_id] = (1.0, 1.0)
        elif a_id in all_unknown_ids:
            arr = np.array(post_assignments[a_id], dtype=int)  # 0=clearancey,1=active
            # We'll interpret P(clearancey) = fraction of 0
            # => active fraction = arr.mean(), so clearancey fraction = 1 - that
            def draw_p_clearancey():
                b = np.random.choice(arr, size=len(arr), replace=True)
                return 1 - b.mean()

            reps = [draw_p_clearancey() for _ in range(n_bootstrap)]
            lower = np.percentile(reps, (1 - confidence_level)/2*100)
            upper = np.percentile(reps, (1 + confidence_level)/2*100)
            ci_intervals[a_id] = (lower, upper)
        else:
            # Agents not in posGroups => no new info => keep prior as both ends
            ci_intervals[a_id] = (p_inf, p_inf)

    # ------------------------------------------------------------
    # 7) Prepare final agentDict
    # ------------------------------------------------------------
    agentDict = {}
    for (a_id, util, p_inf) in agents:
        if a_id in negAgents:
            agentDict[a_id] = (0, 1.0, (1.0, 1.0))
        elif a_id in all_unknown_ids:
            agentDict[a_id] = (util, final_probs[a_id], ci_intervals[a_id])
        else:
            agentDict[a_id] = (util, p_inf, ci_intervals[a_id])

    return agentDict

def solveStaticOverlap(agents, G = G, B = B):

  def generate_overlapping_subsets(agents, G, B):

    # Generate all possible subsets of agents with sizes up to G
    subsets = []
    for r in range(1, G+1):
        subsets.extend(combinations(agents, r))

    # Generate combinations of B subsets without enforcing overlap
    combos = combinations(subsets, B)
    return combos

  def groupHelp(group, posGroups = frozenset(), negAgents = frozenset()):

    groupIDs = frozenset({person[0] for person in group})
    groupUtility = 0
    groupClearancey = 1
    for posGroup in posGroups:
      if posGroup.issubset(groupIDs):
        groupClearancey = 0

    agentDict = bayesTheorem(agents, posGroups, negAgents)

    for person in group:
      groupUtility += agentDict[person[0]][0]
      groupClearancey *= agentDict[person[0]][1]

    groupUtility *= groupClearancey
    return groupClearancey, groupUtility


  def strategyUtility(combination, posGroups = frozenset(), negAgents = frozenset()):

    utility = 0

    if len(combination) > 0:

      firstGroup = combination[0]
      groupIDs = frozenset({person[0] for person in firstGroup})

      firstClearancey, firstUtility = groupHelp(firstGroup, posGroups, negAgents)
      utility += firstUtility

      # nonzero_count test
      if firstClearancey < 1:
        posScenario = combination[1:]

        newPosGroups = set(posGroups.copy())

        for posGroup in newPosGroups:
          if groupIDs.issubset(posGroup):
            newPosGroups.remove(posGroup)

        newPosGroups.add(groupIDs.difference(negAgents))

        utility += strategyUtility(posScenario, frozenset(newPosGroups), frozenset(negAgents)) * (1-firstClearancey)

      # zero_count test
      if firstClearancey > 0:

        negScenario = combination[1:]

        newPosGroups = set()
        for posGroup in posGroups:
          newPosGroups.add(posGroup.difference(groupIDs))

        newNegAgents = set(negAgents.copy())
        newNegAgents.update(groupIDs)

        utility += strategyUtility(negScenario, frozenset(newPosGroups), frozenset(newNegAgents)) * firstClearancey

    return utility

  strategy = dict()
  combos = generate_overlapping_subsets(agents, G, B)
  for combination in combos:
    strategy[combination] = strategyUtility(combination)
  optimal = max(strategy, key=strategy.get)
  return optimal, strategy[optimal]

def solveDynamic(agents, G = G, B = B, posGroups = frozenset(), negAgents = frozenset()):

  if B == 0:
    return [], 0

  def generate_subsets(agents, G=G):
      subsets = []
      for r in range(1, G + 1):
          subsets.extend(combinations(agents, r))
      return subsets

  strategy = dict()
  combos = generate_subsets(agents, G)
  for combination in combos:
    utility = 0

    # first test
    firstTest = combination
    firstUtility = 0
    firstClearancey = 1

    # remaining agents
    remaining = [person for person in agents if person not in firstTest]

    # utility, P(Clearancey) of first test
    firstIDs = frozenset({person[0] for person in firstTest})
    if posGroups:
      for posGroup in posGroups:
        if posGroup.issubset(firstIDs):
          firstClearancey = 0
    else:
      posGroups = frozenset()

    agentDict = bayesTheorem(agents, posGroups, negAgents)

    for person in firstTest:
      firstUtility += agentDict[person[0]][0]
      firstClearancey *= agentDict[person[0]][1]
    utility += firstUtility * firstClearancey

    # nonzero_count scenario
    if firstClearancey < 1:

      newPosGroups = set(posGroups.copy())
      newPosGroups.add(firstIDs.difference(negAgents))

      posStrategy, posUtility = solveDynamic(agents, G, B-1, frozenset(newPosGroups), frozenset(negAgents))

      utility += (1- firstClearancey) * posUtility

    else:

      posStrategy = []

    # zero_count scenario

    if remaining and firstClearancey > 0:
      newPosGroups = set()
      for posGroup in posGroups:
        newPosGroups.add(posGroup.difference(firstIDs))

      newNegAgents = set(negAgents.copy())
      newNegAgents.update(firstIDs)


      negStrategy, negUtility = solveDynamic(remaining, G, B-1, frozenset(newPosGroups), frozenset(newNegAgents))
      utility += firstClearancey * negUtility

    else:

      negStrategy = []

    strategy[(tuple(firstTest), tuple(posStrategy), tuple(negStrategy))] = utility

  optimal = max(strategy, key=strategy.get)
  return optimal, strategy[optimal]

def analyzeTree(tree, agents, posGroups = frozenset(), negAgents = frozenset()):

  utility = 0

  # first test
  firstTest, posStrategy, negStrategy = tree
  firstUtility = 0
  firstClearancey = 1

  # remaining agents
  remaining = [person for person in agents if person not in firstTest]

  # utility, P(Clearancey) of first test
  firstIDs = frozenset(agent[0] if isinstance(agent, tuple) else agent for agent in firstTest)

  if posGroups:
    for posGroup in posGroups:
      if posGroup.issubset(firstIDs):
        firstClearancey = 0
  else:
    posGroups = frozenset()

  # for person in firstTest:
  #   firstUtility += agentDict[person][0]
  #   firstClearancey *= agentDict[person][1]
  # utility += firstUtility * firstClearancey

  firstUtility, firstClearancey = bayesTheoremGroup(agents, firstIDs, posGroups, negAgents) #type: ignore
  utility += firstUtility * firstClearancey

  # nonzero_count scenario
  if firstClearancey < 1:

    newPosGroups = set(posGroups.copy())
    newPosGroups.add(firstIDs.difference(negAgents))

    posUtility = analyzeTree(posStrategy, agents, frozenset(newPosGroups), frozenset(negAgents)) if posStrategy else 0

    utility += (1- firstClearancey) * posUtility

  else:

    posStrategy = []

  # zero_count scenario

  if remaining and firstClearancey > 0:
    newPosGroups = set()
    for posGroup in posGroups:
      newPosGroups.add(posGroup.difference(firstIDs))

    newNegAgents = set(negAgents.copy())
    newNegAgents.update(firstIDs)


    negUtility = analyzeTree(negStrategy, agents, frozenset(newPosGroups), frozenset(newNegAgents)) if negStrategy else 0
    utility += firstClearancey * negUtility

  else:

    negStrategy = []

  return utility

# %%
def analyzeTreeGibbs(tree, agents, posGroups = frozenset(), negAgents = frozenset(), confidence_level=0.95):

  utility = 0
  lowUtility = 0
  highUtility = 0

  # first test
  firstTest, posStrategy, negStrategy = tree
  firstUtility = 0
  firstClearancey = 1
  lowFirstClearancey = 1
  highFirstClearancey = 1

  # remaining agents
  remaining = [person for person in agents if person not in firstTest]

  # utility, P(Clearancey) of first test
  firstIDs = frozenset(firstTest)
  if posGroups:
    for posGroup in posGroups:
      if posGroup.issubset(firstIDs):
        firstClearancey = 0
  else:
    posGroups = frozenset()

  agentDict = GibbsMCMCWindow(agents, posGroups, negAgents, confidence_level=confidence_level) 

  for person in firstTest:
    firstUtility += agentDict[person][0]
    firstClearancey *= agentDict[person][1]
    lowFirstClearancey *= agentDict[person][2][0]
    highFirstClearancey *= agentDict[person][2][1]
  utility += firstUtility * firstClearancey
  lowUtility += firstUtility * lowFirstClearancey
  highUtility += firstUtility * highFirstClearancey

  # nonzero_count scenario
  if firstClearancey < 1:

    newPosGroups = set(posGroups.copy())
    newPosGroups.add(firstIDs.difference(negAgents))

    posUtility, (lowPosUtility, highPosUtility) = analyzeTreeGibbs(posStrategy, agents, frozenset(newPosGroups), frozenset(negAgents), confidence_level=confidence_level) if posStrategy else (0, (0, 0))

    utility += (1- firstClearancey) * posUtility
    lowUtility += (1 - lowFirstClearancey) * lowPosUtility
    highUtility += (1 - highFirstClearancey) * highPosUtility

  # zero_count scenario

  if remaining and firstClearancey > 0:
    newPosGroups = set()
    for posGroup in posGroups:
      newPosGroups.add(posGroup.difference(firstIDs))

    newNegAgents = set(negAgents.copy())
    newNegAgents.update(firstIDs)

    negUtility, (lowNegUtility, highNegUtility) = analyzeTreeGibbs(negStrategy, agents, frozenset(newPosGroups), frozenset(newNegAgents), confidence_level=confidence_level) if negStrategy else (0, (0, 0))

    utility += firstClearancey * negUtility
    lowUtility += lowFirstClearancey * lowNegUtility
    highUtility += highFirstClearancey * highNegUtility

  return utility, (lowUtility, highUtility)

# %%
def analyzeTreeDraw(tree, agents, confidence_level=0.95, bootstrap_draws=1000, cumulative_prob = 0.95, max_draws = 100000, unWeight=False, clearanceStatus = None):

    agentDict = dict()

    for (id, utility, clearance) in agents:
        agentDict[id] = (utility, clearance)

    utilities = []
    probabilities = []

    probDict = dict()
    utilDict = dict()

    draws_taken = 0

    while sum(probDict.values()) < cumulative_prob and draws_taken < max_draws:

        draws_taken += 1

        negAgents = set()
        utility = 0
        probability = 1
        negTestedAgents = set()

        for agent in agents:

            id, _, clearance = agent
            # Determine if the agent is clearancey based on their clearance probability
            is_clearancey = clearanceStatus[id] if clearanceStatus else random.random() < clearance

            # If the agent is unclearancey, add to the zero_count_agents list
            if is_clearancey:
                negAgents.add(id)
                probability *= clearance
            else:
                probability *= 1 - clearance

        if frozenset(negAgents) in probDict:
            continue

        def exploreTree(tree):
            test, posScenario, negScenario = tree
            if test.issubset(negAgents):
                negTestedAgents.update(test)
                if negScenario:
                    exploreTree(negScenario)
            elif posScenario:
                exploreTree(posScenario)

        exploreTree(tree)

        probDict[frozenset(negAgents)] = probability

        for id in negTestedAgents:
            utility += agentDict[id][0]

        utilDict[frozenset(negAgents)] = utility

        utilities.append(utility)
        probabilities.append(probability)

    weighted_mean = 0

    orderedProbabilities = []
    orderedUtilities = []

    for instance in probDict:
        weighted_mean += probDict[instance] * utilDict[instance]
        orderedProbabilities.append(probDict[instance])
        orderedUtilities.append(utilDict[instance])

    orderedProbabilities = np.array(orderedProbabilities)
    pathsExplored = orderedProbabilities.sum()
    weighted_mean /= pathsExplored
    orderedProbabilities /= pathsExplored
    
    # Bootstrap redrawing
    bootstrap_means = []
    for _ in range(bootstrap_draws):
        # Redraw with replacement, weighted by normalized probabilities
        drawn_utilities = np.random.choice(orderedUtilities, size=len(utilities), p=orderedProbabilities, replace=True)
        bootstrap_means.append(np.mean(drawn_utilities))
    
    # Calculate confidence intervals
    lower_percentile = (100 - confidence_level * 100) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower, ci_upper = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])

    if unWeight:
        return weighted_mean, (ci_lower, ci_upper), pathsExplored, next(iter(utilDict.values()))
    else:
        return weighted_mean, (ci_lower, ci_upper), pathsExplored


# %% [markdown]
# ### RL Model

# %% [markdown]
# #### Training Model
### adapted from https://github.com/edwinlock/csef/tree/main/optimisation/python

# utility values erroneous from Gibs drawing and only using marginal probabilities
def solveConicGibbsGreedyDynamicCount(agents, G = G, B = B, clearanceStatus = None):

  if B == 0:
    return [], 0
  
  if not clearanceStatus:
    clearanceStatus = [1 if random.random() < clearance else 0 for (_, _, clearance) in agents]

  posGroups = []
  negAgents = set()

  for i in range(B, 0, -1):
    agentDict = GibbsMCMCWindowCount(agents, posGroups, negAgents)
    updatedAgents = [(id, utility, clearance) for id, (utility, clearance, _) in agentDict.items()]
    firstTest, _ = solveConicSingle(updatedAgents, G=G)
    firstIDs = frozenset({person[0] for person in firstTest})
    posCount = sum(not clearanceStatus[id] for (id, _, _) in agents if id in firstIDs)
    if posCount == 0:
      negAgents.update(firstIDs)
    else:
      posGroups.append((firstIDs, posCount))

  utility = sum(util for (id, util, _) in agents if id in negAgents)

  return utility

Gvals = [3]
Bvals = [5]

for G in Gvals:
  for B in Bvals:
    file_path = f"draw_N50_d2_B{B}_G{G}_Utils3.csv"
    df = pd.read_csv(file_path)

    function = solveConicGibbsGreedyDynamicCount

    tqdm.pandas()

    # Add an empty column if not present for storing results
    if function.__name__ not in df.columns:
        df[function.__name__] = None

    from joblib import Parallel, delayed
    from tqdm import tqdm
    import ast
    import pandas as pd
    import numpy as np

    # Define the column name to check for missing values
    output_column = function.__name__

    def process_row(i, agent_data, clearanceStatus):
        val = function(agent_data, G=G, B=B, clearanceStatus=clearanceStatus) # type: ignore
        return i, val # type: ignore
        # return i, analyzeTreeDraw(generate_binary_tree(tree), agent_data, max_draws=1, unWeight=True, clearanceStatus=clearanceStatus)[3] # type: ignore

    batch_size = 10
    n_jobs = -1  # Use all CPU cores

    # Identify rows that need processing
    rows_to_process = df[df[output_column].isna()].index.tolist()

    for batch_start in tqdm(range(0, len(rows_to_process), batch_size), desc="Processing batches", unit="batch"):
        batch_indices = rows_to_process[batch_start: batch_start + batch_size]
        
        row_data = [(i, ast.literal_eval(df.at[i, 'agents']), ast.literal_eval(df.at[i, 'clearanceStatus'])) for i in batch_indices]

        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
            delayed(process_row)(i, data, clearanceStatus) for i, data, clearanceStatus in row_data
        )
        
        # Update DataFrame
        for i, value in results:
            df.at[i, output_column] = value

        df.to_csv(file_path, index=False)
        print(f"Saved at row {batch_start + batch_size}", flush=True)
