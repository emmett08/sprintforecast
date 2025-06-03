# sprintforecast

Hierarchical Bayesian tooling for reliable sprint planning.  
Generates:

* `triads.csv` — PERT `(o,m,p)` estimates extracted from **open** GitHub issues  
* `actuals.csv` — realised effort for **closed** issues  
* `pool.csv` — PERT triads for backlog items (open issues **not yet assigned to a sprint**)  
* Calibrated **forecast** of goal-hit probability, Brier, CRPS and timeline percentiles  
* **Intake plan** (ticket count & size mix) for the next sprint at a chosen risk level

Pure-Python, cross-platform, requires Python ≥ 3.10.

---

## Installation

```bash
pip install sprintforecast 

git clone https://github.com/emmett08/sprintforecast.git
cd sprintforecast
python -m pip install -e ".[dev]"
````

---

## Command-line usage

First export a GitHub personal-access token with `repo` scope:

```bash
export GITHUB_TOKEN=ghp_xxx
```

### 1  Extract PERT triads from open issues

Open issues must carry three labels of the form `o=<hrs>`, `m=<hrs>`, `p=<hrs>`.

```bash
sprintforecast triads my-org my-repo --outfile triads.csv
```

### 2  Extract actual effort from closed issues

```bash
sprintforecast actuals my-org my-repo --outfile actuals.csv
```

### 3  Generate a backlog pool for intake planning

`pool.csv` contains the PERT triads of all backlog items (open issues **without** a sprint milestone).

```bash
sprintforecast pool my-org my-repo --outfile pool.csv
```

### 4  Fit the error distribution

```bash
sprintforecast fit-error triads.csv actuals.csv error.json
```

Automatically chooses Log-Normal or Skew-$t$ by tail-weight test.

### 5  Sprint forecast

```bash
sprintforecast forecast triads.csv capacity.txt error.json --sprint-hours 80
```

`capacity.txt` is a newline-separated list of historic net capacity (hours per sprint).

Example output:

```json
{
  "p50": 63.9,
  "p80": 76.2,
  "p95": 91.8,
  "P_goal": 0.77,
  "Brier": 0.1771,
  "CRPS": 4.86
}
```

### 6  Next-sprint intake plan

```bash
sprintforecast plan-intake pool.csv capacity.txt error.json \
    --hours-per-sprint 80 --gamma 0.8
```

Returns the recommended ticket count and expected XS/S/M/L mix, momentum-adjusted.

---

## Python API

```python
import pandas as pd, numpy as np
from sprintforecast import (
    rng, Pert, choose_error_family,
    CapacityPosterior, SprintForecaster, IntakePlanner
)

triads = pd.read_csv("triads.csv")
actuals = pd.read_csv("actuals.csv")
residuals = np.log(actuals["actual"] / (triads[["o","m","p"]].values @ [1,4,1]/6))
dist = choose_error_family(residuals)

cap_post = CapacityPosterior(mu=4.4, sigma=0.27)
forecaster = SprintForecaster(dist, cap_post, triads)
print(forecaster.summary(80))

pool = pd.read_csv("pool.csv")
planner = IntakePlanner(pool_df=pool, error_dist=dist,
                        next_capacity=cap_post.sample(10_000))
print(planner.plan(gamma=0.8))
```

---

## Continuous integration example

```yaml
# .github/workflows/ci.yml
name: forecast
on: [push]

jobs:
  forecast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.x' }
      - run: |
          pip install sprintforecast
          sprintforecast triads my-org my-repo
          sprintforecast actuals my-org my-repo
          sprintforecast pool   my-org my-repo
          sprintforecast fit-error triads.csv actuals.csv err.json
          sprintforecast forecast triads.csv cap.txt err.json --sprint-hours 80
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Licence

MIT.  See `LICENCE` file.
