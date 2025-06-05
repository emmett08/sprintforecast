# SprintForecast

A hierarchical Bayesian sprint‑forecast model that converts each ticket’s PERT triad into a scaled‑Beta prior, layers skew‑Student‑t execution noise with queue‑based review and capacity terms, and finally Monte‑Carlo‑simulates the Dev → Review → Done flow to yield the sprint‑completion probability and expected carry‑over.

## TODO
- [ ] Implement the hierarchical Bayesian model
- [ ] Add unit tests for the model components
- [ ] Prioritise dependencies of issues in plan
- [ ] Implement the Monte Carlo simulation for sprint completion
- [ ] Visualise the sprint forecast results and carry-over and interpretation of the results
- [ ] Move class and function definitions to separate files for better organisation
- [ ] Refactor cli.py using SOLID principles for better maintainability

*NOTE* that this is a work in progress, and the model is not yet implemented. The code is provided for educational purposes only.