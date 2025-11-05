#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pareto.py - A multi-objective GA
author: Bill Thompson
license: GPL 3
copyright: 2025-10-29

NSGA-II (SBX crossover + polynomial mutation) with a Pareto plot
Help from ChatGPT
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- Basic Parameters ----------
class Problem:
    """
    A class to hold basic GA parameters and evaluation.
    """
    def __init__(self, pop_size: int = 100, n_gen: int = 120, 
                 pc: float = 0.9, eta_c: float = 15.0, 
                 eta_m: float = 20.0, seed: int | None = None,
                 upper: float = 5.0, lower: float = -5.0)-> None:
        self.n_var = 1 
        self._lower = np.array([lower])
        self._upper = np.array([upper])
        self.n_obj = 2
        self.eval = self._eval
        self._pop_size = pop_size   # population size
        self._n_gen = n_gen         # number of GA iterations
        self._pc = pc               # pronbility of crossover
        self._eta_c = eta_c
        self._eta_m = eta_m
        if seed is None:
            self._seed = int(datetime.now().timestamp())
        else:
            self._seed = seed

    @staticmethod
    def _eval(x: np.ndarray) -> np.ndarray:
        xx = x[0]
        return np.array([xx*xx, (xx-2.0)*(xx-2.0)])  # minimize both
    
    @property
    def pop_size(self) -> int:
        return self._pop_size
    
    @property
    def n_gen(self) -> int:
        return self._n_gen
    
    @property
    def pc(self) -> float:
        return self._pc
    
    @property
    def eta_c(self) -> float:
        return self._eta_c
    
    @property
    def eta_m(self) -> float:
        return self._eta_m

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def lower(self) -> np.ndarray:
        return self._lower
    
    @property
    def upper(self) -> np.ndarray:
        return self._upper

# ---------- NSGA-II utilities ----------
def non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    """
    Sort objective values into dominated frons, i.e. collections of solotions not dominated by
    any thing with a lower front index 

    Parameters
    ----------
    F : np.ndarray
        an array of objective values

    Returns
    -------
    list[list[int]]
        array of fronts. fronts[0] are non-dominated, i.e. the Pareto set.
    """
    N = F.shape[0]
    S = [[] for _ in range(N)]      # S[p]: the set of solutions that p dominates.
    n = np.zeros(N, dtype=int)      # n[p]: the number of solutions that dominate p (its “domination count”).
    rank = np.zeros(N, dtype=int)   # which front p belongs to (0 = first/non-dominated).
    fronts = [[]]                   # list of fronts

    for p in range(N):
        for q in range(N):
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                # p dominates q
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                # q dominate p, increment domination count
                n[p] += 1
        if n[p] == 0:
            # nobody dominate p
            rank[p] = 0
            fronts[0].append(p)

    # build the rest of the fronts
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    # eveything that dominated q is in a previous front, put q in next front
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    fronts.pop()    # there's an extra [] at the end, remove it
    return fronts

def crowding_distance(F: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Calculate crowding distance

    Parameters
    ----------
    F : np.ndarray
        solution values
    front : list[int]
        fronts

    Returns
    -------
    np.ndarray
        the crowding distances for each elemnt
    """
    if not front:
        return np.array([])
    m = F.shape[1]
    l = len(front)
    dist = np.zeros(l)
    if l == 1: dist[0] = np.inf; return dist
    if l == 2: return np.array([np.inf, np.inf])

    for j in range(m):
        idx = np.argsort(F[front, j])
        f_sorted = F[front, j][idx]
        min_f, max_f = f_sorted[0], f_sorted[-1]
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        denom = max_f - min_f if max_f > min_f else 1.0
        for k in range(1, l - 1):
            dist[idx[k]] += (f_sorted[k+1] - f_sorted[k-1]) / denom
    return dist

def binary_tournament(pop: np.ndarray, 
                      ranks: np.ndarray, 
                      cdists: np.ndarray, 
                      rng: random.Random) -> int:
    """
    GA selection

    Parameters
    ----------
    pop : np.ndarray
        the population
    ranks : np.ndarray
        rank of each element in fronts
    cdists : np.ndarray
        crowding distnaces
    rng : random.Random
        random number generator

    Returns
    -------
    int
        the selected elemnt
    """
    i, j = rng.randrange(len(pop)), rng.randrange(len(pop))
    if ranks[i] < ranks[j]: return i
    if ranks[j] < ranks[i]: return j
    return i if cdists[i] > cdists[j] else j

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, 
                  lower: np.ndarray, upper: np.ndarray, 
                  eta: float, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """
    SBX crossover
    
    Parameters
    ----------
    p1 : np.ndarray
        child 1
    p2 : np.ndarray
        child 2
    lower : np.ndarray
        lower x bount
    upper : np.ndarray
        upper bound
    eta : float
        crossover parameter
    rng : random.Random
        random number generator

    Returns
    -------
     tuple[np.ndarray, np.ndarray]
        mutated children
    """
    n = p1.size
    c1, c2 = p1.copy(), p2.copy()
    for i in range(n):
        if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
            x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            L, U = lower[i], upper[i]
            r = rng.random()

            beta = 1.0 + (2.0 * (x1 - L) / (x2 - x1))
            alpha = 2.0 - pow(beta, -(eta + 1.0))
            if r <= 1.0/alpha: betaq = pow(r * alpha, 1.0/(eta+1.0))
            else:              betaq = pow(1.0/(2.0 - r*alpha), 1.0/(eta+1.0))
            c1_i = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + (2.0 * (U - x2) / (x2 - x1))
            alpha = 2.0 - pow(beta, -(eta + 1.0))
            if r <= 1.0/alpha: betaq = pow(r * alpha, 1.0/(eta+1.0))
            else:              betaq = pow(1.0/(2.0 - r*alpha), 1.0/(eta+1.0))
            c2_i = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

           # make sure new elemenst stay in range
            c1[i] = np.clip(c1_i, L, U)
            c2[i] = np.clip(c2_i, L, U)
    return c1, c2

def polynomial_mutation(x: np.ndarray, 
                        lower: np.ndarray, upper: np.ndarray, 
                        eta: float, pm: float, 
                        rng: random.Random) -> np.ndarray:
    """
    ake small mutations to a child solution

    Parameters
    ----------
    x : np.ndarray
        a member of the population
    lower : float
        lower x bound
    upper : np.ndarray
        _description_
    eta : np.ndarray
        tweak size
    pm : float
        probability of mutation
    rng : random.Random
        random number generator

    Returns
    -------
    np.ndarray
        a possibly mutated child
    """
    y = x.copy()
    for i in range(x.size):
        if rng.random() < pm:
            L, U = lower[i], upper[i]
            if U - L < 1e-14: continue
            delta1 = (x[i] - L) / (U - L)
            delta2 = (U - x[i]) / (U - L)
            r = rng.random()
            mut_pow = 1.0 / (eta + 1.0)
            if r < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * r + (1.0 - 2.0 * r) * pow(xy, (eta + 1.0))
                deltaq = pow(val, mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * pow(xy, (eta + 1.0))
                deltaq = 1.0 - pow(val, mut_pow)
            y[i] = np.clip(x[i] + deltaq * (U - L), L, U)
    return y

# ---------- NSGA-II ----------
def nsga2(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """
    NSGA-II Non-Dominated Sorting Genetic Algorithm II

    Parameters
    ----------
    problem : Problem
        An object of the class Problem

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        paired X and objective values
    """
    pop_size = problem.n_gen
    n_gen = problem.n_gen
    pop_size = problem.pop_size
    pc = problem.pc
    eta_c = problem.pc
    eta_m = problem.eta_m
    rng = random.Random(problem.seed)
    
    # init
    X = np.array([rng.random() for _ in range(problem.n_var * pop_size)]).reshape(pop_size, problem.n_var)
    X = problem.lower + X * (problem.upper - problem.lower)
    F = np.array([problem.eval(ind) for ind in X])

    for _ in range(n_gen):
        fronts = non_dominated_sort(F)
        ranks = np.full(pop_size, np.inf)
        cdists = np.zeros(pop_size)  # crowding distances
        
        # get the rank, ie. which front it belongs to, for each solution
        for rank, front in enumerate(fronts):
            ranks[np.array(front)] = rank
            cd = crowding_distance(F, front)
            if cd.size: cdists[np.array(front)] = cd

        # GA
        mating_idx = [binary_tournament(X, ranks, cdists, rng) for _ in range(pop_size)]
        mating_pool = X[mating_idx]

        # offspring
        off = []
        for i in range(0, pop_size, 2):
            p1, p2 = mating_pool[i], mating_pool[(i+1) % pop_size]
            if rng.random() < pc:
                c1, c2 = sbx_crossover(p1, p2, problem.lower, problem.upper, eta_c, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            pm = 1.0 / problem.n_var
            c1 = polynomial_mutation(c1, problem.lower, problem.upper, eta_m, pm, rng)
            c2 = polynomial_mutation(c2, problem.lower, problem.upper, eta_m, pm, rng)
            off.append(c1); off.append(c2)
        off = np.array(off[:pop_size])
        F_off = np.array([problem.eval(ind) for ind in off])

        # environmental selection
        # take by fronts.
        Xc = np.vstack([X, off]); 
        Fc = np.vstack([F, F_off])
        fronts = non_dominated_sort(Fc)
        new_X, new_F = [], []
        for front in fronts:
            if len(new_X) + len(front) <= pop_size:
                new_X.extend(list(Xc[front])); 
                new_F.extend(list(Fc[front]))
            else:
                # take best by crowding distance
                cd = crowding_distance(Fc, front)
                order = np.argsort(-cd)
                remain = pop_size - len(new_X)
                chosen = [front[i] for i in order[:remain]]
                new_X.extend(list(Xc[chosen])); new_F.extend(list(Fc[chosen]))
                break
        X, F = np.array(new_X), np.array(new_F)

    return X, F

# ---------- Run & Plot ----------
def main() -> None:
    prob = Problem(pop_size = 100, 
                   n_gen = 120,
                   pc = 0.9, 
                   eta_c = 15.0, 
                   eta_m = 20.0, 
                   seed = None)
    X, F = nsga2(prob)
    
    # take first Pareto front
    fronts = non_dominated_sort(F)
    pf = F[fronts[0]]
    
    pf_eval = np.zeros((X.shape[0], X.shape[1]+1))
    pf_eval[:, :X.shape[1]] = X[fronts[0]]
    pf_eval[:,-1] = np.array([np.sum((prob.eval(xx))) for xx in X[fronts[0]]])
 
    # save Pareto front and X values
    np.savetxt("output/pareto_front.csv", pf, delimiter=",", header="f1,f2j", comments="")
    header = ','.join(['x(' + str(i) +')' for i in range(pf_eval.shape[1]-1)]) + ',obj'
    np.savetxt("output/pareto_decisions.csv", pf_eval, delimiter=",", header=header, comments="")

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(pf[:,0], pf[:,1])
    plt.xlabel("Objective 1 (f1)")
    plt.ylabel("Objective 2 (f2)")
    plt.title(f"Pareto Front (NSGA-II)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
