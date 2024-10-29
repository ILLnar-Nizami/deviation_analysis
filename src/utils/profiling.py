import functools


def profile(func):
    """Basic profiling decorator"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def profile_memory(func):
    """Memory profiling decorator"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def profile_line(func):
    """Line profiling decorator"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None

try:
    from memory_profiler import profile as memory_profile
except ImportError:
    memory_profile = None
