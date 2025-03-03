import jax
import optax


def optaximiser(
    obj,
    thresh=None,
    num_iters=1_000,
    optimizer=optax.adam(learning_rate=1e-2),
    vb=False,
    jit=True,
    vb_interval=100,
):

    val_grad = jax.value_and_grad(obj)

    def step(carry, _):
        theta, opt_state = carry
        loss_value, grads = val_grad(theta)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return (theta, opt_state), loss_value

    if jit:
        step = jax.jit(step)

    if thresh is None and not vb:
        # Use jax scan for speed
        def opt(theta0):
            opt_state = optimizer.init(theta0)
            (theta, opt_state), losses = jax.lax.scan(
                step, (theta0, opt_state), jax.numpy.arange(num_iters)
            )
            return theta, losses

    else:
        # use loop
        def opt(theta0):
            opt_state = optimizer.init(theta0)
            i = 0.0
            old = 1e10
            losses = []
            theta = theta0
            while i <= num_iters:
                (theta, opt_state), loss_value = step((theta, opt_state), None)
                losses.append(loss_value)
                if vb and i % vb_interval == 0:
                    print(f"step {i}, loss: {loss_value}")
                if thresh is not None and 0 < (old - loss_value) < thresh:
                    if vb:
                        print(
                            f"Converged with delta loss of {(old - loss_value):.4g} after {i} iterations"
                        )
                    break
                i += 1
                old = loss_value
            return theta, jax.numpy.array(losses)

    return opt


def best(thetas, histories):
    best = jax.numpy.argmin(histories[:, -1]).block_until_ready()
    return {k: v[best] for k, v in thetas.items()}
