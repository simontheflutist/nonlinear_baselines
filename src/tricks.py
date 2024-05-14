import contextlib
import time


@contextlib.contextmanager
def timer(time_func=time.perf_counter, message="Complete in {:.4g}"):
    t0 = time_func()
    try:
        yield
    finally:
        t1 = time_func()
        elapsed = t1 - t0
        print(message.format(elapsed))


