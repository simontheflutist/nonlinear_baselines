# %%
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor


# ## helpers


def pd2(a, b):
    return ((a[:, None] - b) ** 2).sum(-1)


def thexp(theta):
    return {k: jnp.exp(v) for k, v in theta.items()}


def get_wood(A, U, C, V):
    return


# %% kernels


def Lin(a, b, theta):
    return theta["sf_lin"] * (a[:, None] * b).sum(-1)


def SE(a, b, theta):
    return theta["sf_se"] * jnp.exp(-0.5 * pd2(a, b) / theta["ll"] ** 2)


def Mat23(a, b, theta):
    d = pd2(a, b)**0.5
    t1 = jnp.sqrt(3) * d / theta["ll"] ** 2
    return theta["sf_m32"] * (1 + t1) * jnp.exp(-t1)


def Mat52(a, b, theta):
    d = pd2(a, b)**0.5
    t1 = jnp.sqrt(5) * d / theta["ll"] ** 2
    return theta["sf_m52"] * (1 + t1 + (t1**2)/3) * jnp.exp(-t1)

# %% GP helpers


def GP(X, Y, kernel, mu=lambda x: jnp.zeros((x.shape[0], 1))):
    mux = mu(X)

    @jax.jit
    def train(theta):
        theta = thexp(theta)
        L, _ = cho_factor(
            kernel(X, X, theta) + theta["sn"] * jnp.eye(X.shape[0]),
            overwrite_a=0,
            lower=1,
        )
        alpha = cho_solve((L, 1), Y - mux)
        return {"L": L, "alpha": alpha, "theta": theta}

    @jax.jit
    def predict_f(Xp, state, vp=None):
        Kstar = kernel(X, Xp, state["theta"])
        return mu(Xp) + Kstar.T @ state["alpha"]

    @jax.jit
    def predict(Xp, state, vp=None):
        Kstar = kernel(X, Xp, state["theta"])
        yp = Kstar.T @ state["alpha"]
        v = cho_solve((state["L"], 1), Kstar)
        vp = kernel(Xp, Xp, state["theta"]) - Kstar.T @ v
        return mu(Xp) + yp, vp

    @jax.jit
    def nlml(theta):
        state = train(theta)
        t1 = 0.5 * ((Y - mux).T @ state["alpha"]).reshape(())
        t2 = jnp.log(jnp.diag(state["L"])).sum()
        t3 = 0.5 * Y.shape[0] * jnp.log(2 * jnp.pi)
        return t1 + t2 + t3

    return train, predict_f, predict, nlml


