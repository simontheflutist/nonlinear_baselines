from itertools import chain, combinations_with_replacement

import numpy as np


def multicomb(iterable, r, empty=False):
    s = list(iterable)
    gen = chain.from_iterable(combinations_with_replacement(s, r) for r in range(r + 1))
    if not empty:
        _ = gen.__next__()  # exclude empty set
    return gen


def hermite_e(n):
    coeff = np.zeros((n, n))
    coeff[0, 0] = 1
    coeff[1, 1] = 1
    for i in range(2, n):
        coeff[i, 1:] += coeff[i - 1, :-1]
        coeff[i, :] -= coeff[i - 2]
    return coeff.T


def hermite_p(n):
    coeff = np.zeros((n, n))
    coeff[0, 0] = 1
    coeff[1, 1] = 2
    for i in range(2, n):
        coeff[i, 1:] += coeff[i - 1, :-1]
        coeff[i, :] -= coeff[i - 2]
    return coeff.T


def chebychev1(n):
    coeff = np.zeros((n, n))
    coeff[0, 0] = 1
    coeff[1, 1] = 1
    for i in range(2, n):
        coeff[i, 1:] += 2 * coeff[i - 1, :-1]
        coeff[i, :] -= coeff[i - 2]
    return coeff.T


def chebychev2(n):
    coeff = np.zeros((n, n))
    coeff[0, 0] = 1
    coeff[1, 1] = 2
    for i in range(2, n):
        coeff[i, 1:] += 2 * coeff[i - 1, :-1]
        coeff[i, :] -= coeff[i - 2]
    return coeff.T


def legendre(n):
    # weighting function is 0, 1 on [-1,1]
    coeff = np.zeros((n, n))
    coeff[0, 0] = 1
    coeff[1, 1] = 1
    for i in range(2, n):
        coeff[i, 1:] += (2 * i + 1) * coeff[i - 1, :-1] / (i + 1)
        coeff[i, :] -= i * coeff[i - 2] / (i + 1)
    return coeff.T


def monomial(n):
    return np.eye(n)


def get_basis(x, n, base=monomial):
    C = base(n)
    return (x[:, None] ** np.arange(n)) @ C, C
