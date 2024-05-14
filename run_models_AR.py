# %% imports
import json
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt


import jax
import jax.numpy as jnp
from src import polys
from src.hank import NARXify, predict
from src.jacks import jeep, opt
from src.linalg import stable_least_squares as SLS
from src.metrics import AIC, nmse, rmse
from src.tricks import timer
from sklearn.preprocessing import StandardScaler as SS

from data import benchmarks_tvt

np.random.seed(0)
key = jax.random.PRNGKey(0)
# %%

benchmark = "CT"

# %% Data

train, val, tests = dc(benchmarks_tvt[benchmark])

SSu = SS().fit(train.u.reshape(train.u.shape[0], -1))
SSy = SS().fit(train.y.reshape(train.y.shape[0], -1))
inv = SSy.inverse_transform

for sig in [train, val, *tests]:
    sig.u = SSu.transform(sig.u.reshape(sig.u.shape[0], -1))
    sig.y = SSy.transform(sig.y.reshape(sig.y.shape[0], -1))

# Evaluation function


bench_results = {}


def evaluate(model, tests, lags, F, theta, meta={}):
    print(benchmark, model)
    if "theta" in theta:  # only capture GP hyperparameters here
        theta_ = theta["theta"]
    else:
        theta_ = theta

    bench_results[f"{model}"] = {
        "theta": theta_,
        "meta": meta,
    }

    for i, test in enumerate(tests):
        YOSA, slc = predict(*test, *lags, F, theta, "OSA")
        YMPO, slc = predict(*test, *lags, F, theta, "MPO")

        yy = inv(test.y[slc])
        yosa = inv(YOSA)
        ympo = inv(YMPO)

        for metric in [rmse]:
            OSA = metric(yy, yosa)
            MPO = metric(yy, ympo)

            print(f"OSA {metric.__name__}: {OSA:.5g}")
            print(f"MPO {metric.__name__}: {MPO:.5g}")

            bench_results[f"{model}"][f"test_{i+1}"] = {
                "MPO": ympo,
                "OSA": yosa,
                "MPO_rmse": MPO,
                "OSA_rmse": OSA,
            }

        xt = np.arange(test.y.shape[0]) * test.sampling_time
        plt.figure()
        plt.plot(xt[slc], test.y[slc], label="True")
        plt.plot(xt[slc], YOSA, label="OSA")
        plt.plot(xt[slc], YMPO, label="MPO")
        plt.legend()
        plt.figure()


# %% ARX scan


def F_ARX(H, theta):
    return H @ theta["alpha"]


def ARX_MPO(train, val, lags_x, lags_y):
    lags = lags_x, lags_y
    H, slc_train = NARXify(*train, *lags)
    theta = {"alpha": SLS(H, train.y[slc_train])}
    YMPO, slcp = predict(*val, *lags, F_ARX, theta, "MPO")
    return AIC(inv(val.y[slcp]), inv(YMPO), H.shape[1])


nx = 20
ny = 20
spacings = [1]
best = 10e10
scores = np.zeros((len(spacings), nx, ny))

with timer():
    for k, s in enumerate(spacings):
        for i in range(nx):
            for j in range(ny):
                lags_x = tuple(np.arange(0, i + 1, s))
                lags_y = tuple(np.arange(1, j + 2, s))

                MPO = ARX_MPO(train, val, lags_x, lags_y)
                scores[k, i, j] = MPO
                if MPO < best:
                    best = MPO
                    lags = lags_x, lags_y

# %%


# lags = 10,10 # CED uses this structure
# lags = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
H, slc = NARXify(*train, *lags)
print(lags)

bench_results["LSO"] = {
    "lags": lags,
    "nx": nx,
    "ny": ny,
    "spacings": spacings,
    "valscores": scores,
}


# %% ARX

with timer():
    # train
    theta = {"alpha": SLS(H, train.y[slc])}

    evaluate("ARX", tests, lags, F_ARX, theta)

# %% polynomial NARX

orders = np.arange(2, 7 + 1)
scores = []

with timer():
    best = 1e10
    for order in orders:

        def F(H, theta):
            return basis(H, order) @ theta["alpha"]

        def basis(H, order):
            basis = H[..., None] ** np.arange(1, order + 1)
            basis = basis @ polys.legendre(order)
            return basis.reshape(H.shape[0], -1)

        Phi = basis(H, order)
        theta = {"alpha": SLS(Phi, train.y[slc])}
        YMPO, slcp = predict(*val, *lags, F, theta, "MPO")
        if any(np.isnan(YMPO)):
            continue
        if any(np.isinf(YMPO)):
            continue
        score = AIC(inv(val.y[slcp]), inv(YMPO), Phi.shape[1])

        scores.append(score)
        if score < best:
            best = score
            best_order = order

    order = best_order
    print(f"validation =>order={order}")
    Phi = basis(H, order)
    theta = {"alpha": SLS(Phi, train.y[slc])}

    evaluate(
        "PNARX",
        tests,
        lags,
        F,
        theta,
        meta={
            "order": int(order),
            "orders": orders,
            "valscores": scores,
        },
    )


# %% GP NARX SE

Nu = 200
N_opt_GP = 1000
Nre = 1
key, k1, k2, k3 = jax.random.split(key, 4)

theta0 = {
    "sf_se": jnp.zeros((Nre, 1)),
    "sn": jax.random.uniform(k2, minval=-5, maxval=0, shape=(Nre,1)),
    "ll": jax.random.uniform(k3, minval=-3, maxval=2, shape=(Nre,1)),
}


with timer():
    if Nu < H.shape[0]:  # use sparse GP
        idx = np.arange(0, H.shape[0], H.shape[0] // Nu)
        trn, F, _, nlml = jeep.FITC(H, train.y[slc], H[idx], jeep.SE)
    else:
        trn, F, _, nlml = jeep.GP(H, train.y[slc], jeep.SE)

    thetas, hists = jax.pmap(opt.optaximiser(nlml, num_iters=N_opt_GP, jit=1))(theta0)
    
    # xvalidate on val set
    scores = []
    for i in range(Nre):
        theta = {k:v[i] for k,v in thetas.items()}
        state = trn(theta)
        YMPO, slcp = predict(*val, *lags, F, state, "MPO")
        score = rmse(inv(val.y[slcp]), inv(YMPO))
        scores.append(score)
    
    theta = {k:v[np.argmin(scores)] for k,v in thetas.items()}
    state = trn(theta)  # get GP state with opt hyperameters

    plt.plot(hists.T)

    evaluate(
        "GPNARX",
        tests,
        lags,
        F,
        state,
        meta={
            "Kernel": "SE",
            "Mu": "zero",
            "Num_opt_iters": N_opt_GP,
            "Nu": Nu,
            "theta0": theta0,
            "Inducingselection": "UniformGrid",
        },
    )


# %% MLP NARX

Nre = 5
N_opt_MLP = 20_000

with timer():

    def F(H, theta):
        h1 = jnp.tanh(H @ theta["w1"] + theta["b1"])
        return h1 @ theta["w2"] + theta["b2"]

    nhs = [
        2,
        5,
        7,
        10,
    ]
    scores, thetas = [], []

    def obj(theta):
        Yhat, slcp = predict(*train, *lags, F, theta, "OSA")
        return nmse(train.y[slcp], Yhat)

    def trn(key, nh):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        theta0 = {
            "w1": jax.random.uniform(
                k1, minval=-1, maxval=1, shape=(Nre, H.shape[1], nh)
            ),
            "b1": jax.random.uniform(k2, minval=-1, maxval=1, shape=(Nre, nh)),
            "w2": jax.random.uniform(k3, minval=-1, maxval=1, shape=(Nre, nh, 1)),
            "b2": jax.random.uniform(
                k4,
                minval=-1,
                maxval=1,
                shape=(
                    Nre,
                    1,
                ),
            ),
        }
        return jax.pmap(opt.optaximiser(obj, num_iters=N_opt_GP, jit=1))(theta0)

    plt.figure()
    for nh in nhs:
        key, k1 = jax.random.split(key, 2)
        theta_s, hists = trn(k1, nh)
        theta = opt.best(theta_s, hists)
        plt.semilogy(hists.T)        
        Yhat, slcp = predict(*val, *lags, F, theta, "MPO")
        scores.append(nmse(inv(val.y[slcp]), inv(Yhat)))
        print(nh, scores[-1])
        thetas.append(theta)

    # plt.plot(nhs, scores)
    nh = nhs[np.argmin(scores)]
    theta = thetas[np.argmin(scores)]

    evaluate(
        "MLP_NARX",
        tests,
        lags,
        F,
        theta,
        meta={
            "Activation": "tanh",
            "hidden_nodes": nhs,
            "num_hidden": nh,
            "Num_opt_iters": N_opt_MLP,
            "valscores": scores,
            "PRNGKey": key,
        },
    )

# %% save data


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, jnp.ndarray):
            return np.array(obj).tolist()
        elif isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


with open(f"./data/{benchmark}_AR.json", "w") as f:
    json.dump(bench_results, f, cls=Encoder)
