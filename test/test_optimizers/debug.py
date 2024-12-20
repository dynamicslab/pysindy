import pickle
from functools import wraps


def foo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Called {func}")
        return func(*args, **kwargs)

    return wrapper


@foo
def bar(a, b):
    print(a + b)


bars = pickle.dumps(bar)
barl = pickle.loads(bars)
