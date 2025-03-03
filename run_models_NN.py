# %%
import json
import os
from copy import deepcopy as dc

import jax
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.linen.module import nowrap
from jax import numpy as jnp
from sklearn.preprocessing import MinMaxScaler as MM

import data
from src.hank import NARXify
from src.jacks import opt
from src.metrics import rmse
from src.tricks import timer

key = jax.random.PRNGKey(0)

# %% Basic RNN cell (not in linen for some reason?)


class RNNCell(nn.RNNCellBase):
    features: int

    @nn.compact
    def __call__(self, carry, x):
        h = carry
        hidden_features = h.shape[-1]

        dense_h = nn.Dense(hidden_features)
        dense_i = nn.Dense(hidden_features)

        new_h = nn.tanh(dense_i(x) + dense_h(h))
        return new_h, new_h

    @nowrap
    def initialize_carry(self, rng, input_shape):
        batch_dims = input_shape[:-1]
        _, k2 = jax.random.split(rng)
        mem_shape = batch_dims + (self.features,)
        h = nn.initializers.zeros_init()(k2, mem_shape, jnp.float32)
        return h

    @property
    def num_feature_axes(self) -> int:
        return 1


# %% Hyperopt plan

# 'layer_type'    : vary {RNN, GRU, LSTM, OLSTM}
# 'n_opt'         : fix (per benchmark)
# 'batch_len'     : fix (per benchmark)
# 'learn_rate'    : fix (all) 1e-3
# 'n_re'          : fix (all) 10
# 'h_h'           : xval [2,4,8,16]
# 'n_lookback'    : xval [2,4,8,16]

models = {
    "GRU": nn.GRUCell,
    "RNN": RNNCell,
    "LSTM": nn.LSTMCell,
    "OLSTM": nn.OptimizedLSTMCell,
    "FIR_MLP": None,
}

# %% Set or infer from batch job arrays

# if SLURM job
# AR_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
# n = len(models)
# i, j = AR_ID // n, AR_ID % n
# benchmark = list(data.benchmarks_tvt.keys())[i]
# model = list(models)[j]

benchmark = "CT"
model = "RNN"

print(benchmark, model)
# %% CONFIG
# Not pretty but at least explicit


config = {
    "LR": 1e-3,
    "n_re": 10,
    "n_hs": [2, 4, 8, 16, 32],
    "n_look": [2, 4, 8, 16, 32],
}
#   "n_re": 1,       # testing
#   "n_hs": [8],     # testing
#   "n_look": [8]}  # testing

opts = {
    "CED1": {
        "n_opt": 20_000,
        "batch_len": None,
    },
    "CED2": {
        "n_opt": 20_000,
        "batch_len": None,
    },
    "CT": {
        "n_opt": 20_000,
        "batch_len": None,
    },
    "EMPS": {
        "n_opt": 10_000,
        "batch_len": 2000,
    },
    "WH": {
        "n_opt": 10_000,
        "batch_len": 200,
    },
    "SB": {
        "n_opt": 10_000,
        "batch_len": 200,
    },
}

config |= opts[benchmark]

# %% Scale Data onto [-1, 1]

train, val, tests = dc(data.benchmarks_tvt[benchmark])

SSu = MM().fit(train.u.reshape(train.u.shape[0], -1))
SSy = MM().fit(train.y.reshape(train.y.shape[0], -1))
inv = SSy.inverse_transform

for sig in [train, val, *tests]:
    sig.u = SSu.transform(sig.u.reshape(sig.u.shape[0], -1))
    sig.y = SSy.transform(sig.y.reshape(sig.y.shape[0], -1))


# %% Batch data for training


def batch_data(dat, batch_len=None, lookback=20):
    H, slc = NARXify(dat.u, dat.u, lookback, 0)
    N = H.shape[0]
    if batch_len is None:
        batch_len = N
    n_batch = int(jnp.floor(N / batch_len))
    H = H.astype(jnp.float32)
    Hflat = H[None]
    Yflat = dat.y[None, slc]
    Hbatch = H[: n_batch * batch_len].reshape(n_batch, batch_len, lookback)
    Ybatch = dat.y[slc][: n_batch * batch_len].reshape(n_batch, batch_len, 1)
    print(Hbatch.shape)
    return (Hflat, Yflat), (Hbatch, Ybatch)  # all (batch, time, features)


def get_network_trainer(model, X_train, y_train):
    @jax.pmap
    def train_network(key):
        theta0 = model.init(key, X_train)

        @jax.jit
        def mse(params):
            y_hat = model.apply(params, X_train)
            return ((y_train.squeeze() - y_hat.squeeze()) ** 2).sum()

        Optim = opt.optaximiser(
            mse,
            num_iters=config["n_opt"],
            optimizer=optax.adam(learning_rate=config["LR"]),
        )

        return Optim(theta0)

    return train_network


def predict(params, model, dataset, look):
    (X_test, y_test), _ = batch_data(dataset, config["batch_len"], look)
    yhat = model.apply(params, X_test).squeeze()
    ytrue = y_test.squeeze()
    return ytrue[:, None], yhat[:, None]


def multi_predict(params, model, dataset, look):
    (X_test, y_test), _ = batch_data(dataset, config["batch_len"], look)

    @jax.pmap
    @jax.jit
    def prd(prms):
        yhat = model.apply(prms, X_test).squeeze()
        ytrue = y_test.squeeze()
        return ytrue[:, None], yhat[:, None]

    return prd(params)


# %% Xval loop

best = 1e10
for look in config["n_look"]:
    # Batch data
    _, (X_train, y_train) = batch_data(train, config["batch_len"], look)

    for nh in config["n_hs"]:

        if model == "FIR_MLP":

            class network(nn.Module):
                @nn.compact
                def __call__(self, x):
                    x = nn.Dense(nh)(x)
                    x = nn.tanh(x)
                    x = nn.Dense(1)(x)
                    return x

        else:
            # Define model
            class network(nn.Module):
                @nn.compact
                def __call__(self, x):
                    x = nn.RNN(models[model](nh))(x)
                    x = nn.Dense(1)(x)
                    return x

        try:
            # init model
            test_model = network()
            # train with n_re random restarts
            key, k1 = jax.random.split(key)
            trainer = get_network_trainer(test_model, X_train, y_train)
            with timer():
                thetas, histories = trainer(jax.random.split(k1, config["n_re"]))
            # predict on val set and score rmse
            ytrues, yhats = multi_predict(thetas, test_model, val, look)
            scores = jnp.array([rmse(inv(a), inv(b)) for a, b in zip(ytrues, yhats)])
            print(scores)
            # update best on val data
            if (new_best := jnp.min(scores)) < best:
                best = new_best
                best_idx = jnp.argmin(scores)
                theta_star = theta_star = jax.tree_map(lambda a: a[best_idx], thetas)
                xval = [best, nh, look, theta_star, test_model]
                print(best, nh, look)

        except ValueError:  # NaN, inf etc.
            continue
# %% Evaluate best xval model

best, nh, look, theta_star, xval_model = xval

out = {
    "model": model,
    "benchmark": benchmark,
    "meta": config,
    "nh_xval": nh,
    "look_xval": look,
    "theta": jax.tree_map(lambda x: x.tolist(), theta_star),
}

for i, tst in enumerate(tests):

    ytrue, yhat = [inv(a) for a in predict(theta_star, xval_model, tst, look)]

    plt.figure()
    plt.plot(ytrue)
    plt.plot(yhat)
    plt.plot(ytrue - yhat)

    RMSE = rmse(ytrue, yhat)

    out[f"test_{i+1}"] = {"yhat": yhat.squeeze().tolist(), "rmse": RMSE}

    print(f"{benchmark} {model} test_{i+1} {RMSE:.3g}")

# %% Save the results


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()


with open(f"./data/{benchmark}_{model}_NN.json", "w") as f:
    json.dump(out, f, cls=Encoder)
