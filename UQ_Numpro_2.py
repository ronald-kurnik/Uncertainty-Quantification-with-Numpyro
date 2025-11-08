# --------------------------------------------------------------
#  Bayesian Linear Regression – FINAL, CLEAN, NO ERRORS
# --------------------------------------------------------------

import numpy as np
import jax
import jax.random as random
import numpyro
import numpyro.distributions as dist
import arviz as az
import matplotlib.pyplot as plt

# -------------------------- data --------------------------------
np.random.seed(0)
X = np.linspace(-3, 3, 20)                     # 1-D
true_w, true_b = 1.5, -0.5
y = true_w * X + true_b + np.random.normal(0, 0.5, X.shape[0])

# -------------------------- model ------------------------------
def model(X, y=None):
    w     = numpyro.sample("w",     dist.Normal(0.0, 2.0))
    b     = numpyro.sample("b",     dist.Normal(0.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    mu = w * X + b
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

# -------------------------- inference ---------------------------
num_chains = 4
num_warmup = 500
num_samples = 1000

sampler = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(
    sampler,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True,
)

rng_keys = random.split(random.key(0), num_chains)
mcmc.run(rng_keys, X=X, y=y)

# -------------------------- diagnostics ------------------------
idata = az.from_numpyro(mcmc)
print("\n=== Parameter Summary ===")
print(az.summary(idata, var_names=["w", "b", "sigma"]))

# -------------------------- posterior predictive ---------------
# Flatten posterior samples: (4000,) instead of (4, 1000)
post = mcmc.get_samples(group_by_chain=True)
posterior_samples = {k: v.reshape(-1) for k, v in post.items()}

# Create predictive distribution
predictive = numpyro.infer.Predictive(model, posterior_samples=posterior_samples)

# CORRECT: positional args → model(X, y=None)
X_new = np.linspace(-4, 4, 400)
y_pred = predictive(random.key(1), X_new, y=None)["obs"]  # (4000, 400)

# -------------------------- plot --------------------------------
mean  = y_pred.mean(axis=0)
lower = np.percentile(y_pred, 2.5, axis=0)
upper = np.percentile(y_pred, 97.5, axis=0)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="black", label="data", zorder=5)
plt.plot(X_new, mean, color="#1f77b4", lw=2, label="posterior mean")
plt.fill_between(X_new, lower, upper, color="#1f77b4", alpha=0.3, label="95% CI")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Bayesian Linear Regression – Uncertainty Quantification")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()