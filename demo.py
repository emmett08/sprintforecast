import pathlib, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sprintforecast import (
    Pert, CapacityPosterior, SprintForecaster, MomentumModel,
    IntakePlanner, LogNormalError, SkewTError
)

triads = pd.read_csv("triads.csv").astype(float)
cap_hist = np.loadtxt("capacity_hist.txt")
dist = {"lognormal": LogNormalError.from_dict,
        "skewt":     SkewTError.from_dict}[json.loads(pathlib.Path("dist.json").read_text())["type"]](json.loads(pathlib.Path("dist.json").read_text()))
cap_post = CapacityPosterior(*((lambda x: (x.mean(), x.std(ddof=1)))(np.log(cap_hist))))
forecaster = SprintForecaster(dist, cap_post, triads)
t_complete = forecaster._simulate()

fig, ax = plt.subplots()
ax.hist(t_complete, bins=40, density=True)
ax.set_xlabel("completion hours")
ax.set_ylabel("density")
fig.tight_layout()

fig, ax = plt.subplots()
ecdf_x = np.sort(t_complete)
ecdf_y = np.arange(1, len(ecdf_x)+1) / len(ecdf_x)
ax.step(ecdf_x, ecdf_y)
ax.axvline(80, linestyle="--")
ax.set_xlabel("hours")
ax.set_ylabel("P(T ≤ t)")
fig.tight_layout()

momentum = MomentumModel(cap_hist, 80)
planner = IntakePlanner(triads, dist, momentum.forecast(10_000))
res = planner.plan(0.8)
sizes = res["size_mix"]
fig, ax = plt.subplots()
ax.bar(sizes.keys(), sizes.values())
ax.set_ylabel("proportion")
fig.tight_layout()

plt.show()
