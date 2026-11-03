import pytest


@pytest.mark.slow
def test_full_healing_loop():
    from selfheal import e2e
    events = e2e.run(verbose=False)
    assert events["new"] != events["old"]
