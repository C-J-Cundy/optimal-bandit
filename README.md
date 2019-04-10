# Implementing Bandit Convex Optimization
## Repo for an implementation of the bandit convex optimization algorithm in [Hazan and Levy]

The bandit convex optimization algorithm in [Hazan and Levy] doesn't have any implementations
on the web, despite having a fairly simple form.

It is the first algorithm to obtain O(sqrt(T)) regret, matching the lower bound given in
[Shamir]. It is a bandit (i.e. zero-order) algorithm, i.e. it doesn't use any gradient information.
It relies on a trick to approximate the gradient by sampling in a sphere around the
current iterate.