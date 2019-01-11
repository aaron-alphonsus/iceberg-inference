# Dartmouth REU 2018
This is a project in collaboration with CRREL to infer iceberg drag coefficients. Part of a larger effort to track and predict iceberg locations in the North Atlantic.

Our approach is to utilize Bayesian inference to quantify the uncertainty in our inferred coefficients. The Bayesian paradigm uses probability distributions to model our state of knowledge about the uncertain frictional coefficients before and after observing the iceberg's path. We use a Markov chain Monte Carlo (MCMC) algorithm to generate correlated samples from the posterior distribution and use these samples to compute a Monte Carlo estimate of the expected coefficients.
