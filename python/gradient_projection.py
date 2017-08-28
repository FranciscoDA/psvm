#! /usr/bin/env python3

import math
from functools import reduce
from statistics import mean

class Example:
    def __init__(self, x_i, y_i):
        self.x = x_i
        self.y = y_i

epsilon = 0.0001

# training dataset
S = [
    Example([-1, 10], 1), Example([0,9], 1), Example([1,10], 1),
    Example([-1,8], -1), Example([0,8], -1), Example([1,8], -1),
]
n = len(S)
d = len(S[0].x)

# starting point = 0
alpha = [0.0]*n

# working set (empty)
B = set()
# C hyperparameter
C = 10

def approx_gt(a,b):
    return a >= b - epsilon
def approx_lt(a,b):
    return a <= b + epsilon
def approx_eq(a,b):
    return approx_gt(a,b) and approx_lt(a,b)

# vector operations
def scalar(s, a):
    "Scalar product between scalar s and vector a"
    return tuple(s*a_i for a_i in a)
def dot(a, b):
    "Dot product between column a and row b"
    return sum(a_i * b_i for a_i, b_i in zip(a,b))
def vsum(*summands):
    "Vector summation from summands"
    return reduce(lambda u,v: tuple(v_i+u_i for v_i, u_i in zip(v, u)), summands)

def decision(x, w, b):
    "Linear classifier decision function: w.x+b"
    return dot(x, w) + b
def calc_w(S, alpha):
    "Obtains the omega vector from a dual parameter set"
    return vsum(*(scalar(alpha_i * p.y, p.x) for alpha_i, p in zip(alpha, S)))
def calc_b(S, alpha, w):
    "Obtains the bias from the omega vector and a support vector"
    sv = next(support_vectors(S, C, alpha))
    return sv.y - dot(w, sv.x)
def K(a, b): # positive definite matrix such that K_ij = x_i . x_j or some kernel function
    return dot(a, b)

# KKT conditions for soft-margin
def hingeloss(p, w, b):
    return max(0, 1 - p.y*(dot(w, p.x) + b))
def primal_feasible(p, w, b):
    "Verifies primal feasibility condition for x"
    return approx_gt(p.y * decision(p.x, w, b) - 1 + hingeloss(p, w, b), 0)
def dual_feasible(C, alpha_i):
    "Verifies dual feasibility condition for alpha_i"
    return approx_lt(0, alpha_i) and approx_lt(alpha_i, C)
def complementary_slackness(p, alpha_i, w, b):
    "Verifies complementary slackness condition for x, alpha_i"
    return approx_eq(alpha_i * (p.y * decision(p.x, w, b) - 1 + hingeloss(p, w, b)), 0)
def support_vectors(S, C, alpha):
    "Gets an iterator of the support vectors in S according to alpha"
    return (p for alpha_i, p in zip(alpha, S) if 0 < alpha_i and alpha_i < C)
def is_optimal(C, x, alpha_i, w, b):
    #return dual_feasible(C, alpha_i)
    return dual_feasible(C, alpha_i) and primal_feasible(x, w, b) and complementary_slackness(x, alpha_i, w, b)
def get_violating_samples(C, S, alpha):
    w = calc_w(S, alpha)
    try:
        b = calc_b(S, alpha, w)
        return [i for i,(x,alpha_i) in enumerate(zip(S, alpha)) if not is_optimal(C, x, alpha_i, w, b)]
    except StopIteration:
        # if no support vectors are found, all the samples are violating the optimality
        return list(range(0,n))

def dual_differential(S, alpha, a): # dD/dalpha(alpha_x)
    return 1 - a.y * sum(b.y * alpha_b * K(a.x, b.x) for b, alpha_b in zip(S, alpha))
def dual_hessian(a, b): # Hessian: dD/d(dalpha_a,alpha_b)
    return a.y * b.y * K(a.x, b.x)
def lambda_plus(S, g, u): # optimal step size (unconstrained)
    gu = dot(g,u)
    uhu = sum(u_i * u_j * dual_hessian(S_i, S_j) for S_i, u_i in zip(S,u) for S_j, u_j in zip(S,u))
    return gu/uhu
def lambda_max(C, alpha, u):
    return min(
        max(0, C/u_i) - alpha_i/u_i if u_i != 0 else math.inf
        for alpha_i, u_i in zip(alpha, u)
    )

if __name__ == '__main__':
    g = [0.0]*n
    u = [0.0]*n
    violating_samples = get_violating_samples(C, S, alpha)
    while violating_samples: # while there are examples violating the optimality condition
        B |= set(violating_samples)
        while True:
            for k in B:
                g[k] = dual_differential(S, alpha, S[k])
            while True: # projection-eviction loop
                rho = mean(S[i].y * g[i] for i in B)
                u = [g[k] - p.y*rho if k in B else 0 for k, p in enumerate(S)]
                eviction_set = {k for k in B if (u[k] > 0 and alpha[k] >= C) or (u[k] < 0 and alpha[k] <= 0)}
                if not eviction_set:
                    break
                B -= eviction_set
            if all(approx_eq(u_k, 0) for u_k in u): # all u = 0 implies no direction was found
                break
            lambda_star = max(0, min(lambda_plus(S,g,u), lambda_max(C, alpha, u)))
            print('l* = %f' % lambda_star)
            alpha_new = vsum(alpha, scalar(lambda_star, u))
            #if lambda_star != 0:
            #print('%s + %f * %s = %s' % (alpha, lambda_star, u, alpha_new))
            alpha = alpha_new

        violating_samples = get_violating_samples(C, S, alpha)
        #print(violating_samples)
        print(alpha)
        w = calc_w(S, alpha)
        print('%s + %f' % (w, calc_b(S, alpha, w)))
