import time
from src.main import LatencyProfiler


def test_profiler_init():
    p = LatencyProfiler(window=30)
    assert p.window == 30
    assert len(p.times) == 0
    assert p.total_avg() == 0.0


def test_profiler_mark():
    p = LatencyProfiler()
    p.mark("detect")
    time.sleep(0.001)
    p.mark("detect")
    assert p.avg("detect") > 0.0


def test_profiler_mark_unbalanced():
    p = LatencyProfiler()
    p.mark("detect")
    assert p.avg("detect") == 0.0


def test_profiler_multiple_marks():
    p = LatencyProfiler()
    for _ in range(5):
        p.mark("compute")
        time.sleep(0.0005)
        p.mark("compute")
    assert p.avg("compute") > 0.0


def test_profiler_rolling_average():
    p = LatencyProfiler(window=3)
    for _ in range(10):
        p.mark("test")
        time.sleep(0.001)
        p.mark("test")
    assert len(p.times["test"]) <= 3


def test_profiler_snapshot():
    p = LatencyProfiler()
    p.mark("a")
    time.sleep(0.001)
    p.mark("a")
    p.mark("b")
    time.sleep(0.002)
    p.mark("b")
    snap = p.snapshot()
    assert "a" in snap
    assert "b" in snap
    assert snap["a"] > 0.0
    assert snap["b"] > 0.0


def test_profiler_total():
    p = LatencyProfiler()
    p.mark("x")
    time.sleep(0.001)
    p.mark("x")
    total = p.total_avg()
    assert total > 0.0
    assert total == p.avg("x")


def test_profiler_avg_empty():
    p = LatencyProfiler()
    assert p.avg("nonexistent") == 0.0


if __name__ == "__main__":
    test_profiler_init()
    test_profiler_mark()
    test_profiler_mark_unbalanced()
    test_profiler_multiple_marks()
    test_profiler_rolling_average()
    test_profiler_snapshot()
    test_profiler_total()
    test_profiler_avg_empty()
    print("\nAll profiler tests passed!")
