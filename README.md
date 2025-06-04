# SprintForecast

A hierarchical Bayesian sprint‑forecast model that converts each ticket’s PERT triad into a scaled‑Beta prior, layers skew‑Student‑t execution noise with queue‑based review and capacity terms, and finally Monte‑Carlo‑simulates the Dev → Review → Done flow to yield the sprint‑completion probability and expected carry‑over.

## Usage
```shell
sprintforecast plan \
  --owner emmett08 --repo sprintforecast --team 3 --length 10 --project 8
```

*NOTE* that this is a work in progress, and the model is not yet implemented. The code is provided for educational purposes only.