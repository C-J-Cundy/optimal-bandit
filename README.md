# Implementing Bandit Convex Optimization
## Repo for an implementation of the bandit convex optimization algorithm in [Hazan and Levy 2014]

The bandit convex optimization algorithm in [Hazan and Levy 2014] doesn't have any implementations
on the web, despite having a fairly simple form.

It is the first algorithm to obtain O(sqrt(T)) regret, matching the lower bound given in
[Shamir 2013]. It is a bandit (i.e. zero-order) algorithm, i.e. it doesn't use any gradient information.
It relies on a trick to approximate the gradient by sampling in a sphere around the
current iterate.

We use Jax to evaluate the Hessian without writing it in by hand, and also to get a compact Newton's
method for the inner loop. 



### References
Elad Hazan & Kfir Y. Levy, Bandit Convex Optimization: Towards Tight Bounds. Neurips 2014.

Ohad Shamir, On the complexity of bandit and derivative-free stochastic convex optimization JMLR 2013.

