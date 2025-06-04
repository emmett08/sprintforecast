# SprintForecast

A hierarchical Bayesian sprint‑forecast model that converts each ticket’s PERT triad into a scaled‑Beta prior, layers skew‑Student‑t execution noise with queue‑based review and capacity terms, and finally Monte‑Carlo‑simulates the Dev → Review → Done flow to yield the sprint‑completion probability and expected carry‑over.
