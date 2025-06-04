import numpy as np
import pandas as pd
import sprintforecast as sf
import sprintforecast.pert as pert
import sprintforecast.capacity as cap


def test_summary_deterministic():
    # deterministically seed rng used across modules
    g = np.random.default_rng(0)
    sf._rng.rng = g
    pert.rng = g
    cap.rng = g

    triads = pd.DataFrame({
        "o": [1.0, 2.0, 1.5],
        "m": [2.0, 3.0, 2.5],
        "p": [3.0, 4.0, 3.5],
    })

    error = sf.LogNormalError(mu=0.0, sigma=0.0)
    capacity = sf.CapacityPosterior(mu=0.0, sigma=0.0)

    forecaster = sf.SprintForecaster(error, capacity, triads, paths=1000)
    summary = forecaster.summary(4.0)

    assert 0.0 <= summary["P_goal"] <= 1.0
    assert summary["p50"] <= summary["p80"] <= summary["p95"]
