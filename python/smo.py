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
    Example([-1,10], 1), Example([0,9], 1), Example([1,10], 1),
    Example([-1,8], -1), Example([0,8], -1), Example([1,8], -1),
]
n = len(S)
d = len(S[0].x)

# starting point = 0
alpha = [0.0]*n

# working set (empty)
B = set()
# C hyperparameter
C = 5

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
    "Obtains the bias from the omega vector"
    sv = next(support_vectors(S, alpha))
    return sv.y - dot(w, sv.x)
def K(a, b): # positive definite matrix such that K_ij = x_i . x_j or some kernel function
    return dot(a, b)

# KKT conditions for soft-margin
def hingeloss(p, w, b):
    return max(0, 1 - p.y*(dot(w, p.x) + b))

def support_vectors(S, alpha):
    "Gets an iterator of the support vectors in S according to alpha"
    return (p for alpha_i, p in zip(alpha, S) if alpha_i > 0)

def get_A(ex):
    return 0 if ex.y > 0 else -C
def get_B(ex):
    return C if ex.y > 0 else 0

if __name__ == '__main__':
    g = [1.0]*n
    while True:
        i = max(
            (i for i,p in enumerate(S) if p.y*alpha[i] < get_B(p)),
            key=lambda i: S[i].y*g[i]
        )
        j = min(
            (j for j,p in enumerate(S) if get_A(p) < p.y*alpha[j]),
            key=lambda j: S[j].y*g[j]
        )
        criterion_i, criterion_j = S[i].y*g[i], S[j].y*g[j]
        if criterion_i <= criterion_j:
            break
        Bi, Aj = get_B(S[i]), get_A(S[j])
        Kii, Kij, Kjj = K(S[i].x, S[i].x), K(S[i].x, S[j].x), K(S[j].x, S[j].x)
        lambd = min(
            Bi-S[i].y*alpha[i],
            S[j].y*alpha[j]-Aj,
            (criterion_i-criterion_j)/(Kii + Kjj - 2 * Kij)
        )
        g = [gk + lambd*S[k].y*(K(S[j].x, S[k].x)-K(S[i].x, S[k].x)) for k,gk in enumerate(g)]
        alpha[i] += S[i].y*lambd
        alpha[j] -= S[j].y*lambd
    print(alpha)
