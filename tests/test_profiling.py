import time

from src.utils.profiling import profile, profile_memory, profile_line


@profile
def slow_function():
    time.sleep(1)  # Simulate a slow function


@profile_memory
def memory_intensive_function():
    return [i for i in range(10000)]  # Simulate memory usage


@profile_line
def line_profiler_function():
    return sum(range(1000))  # Simple function to profile


def test_profile_decorator():
    start_time = time.time()
    slow_function()
    end_time = time.time()
    assert end_time - start_time >= 1  # Ensure the function takes time


def test_profile_memory_decorator():
    result = memory_intensive_function()
    assert len(result) == 10000  # Ensure the function returns the expected result


def test_profile_line_decorator():
    result = line_profiler_function()
    assert result == sum(range(1000))  # Ensure the function returns the expected result
