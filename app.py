import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm

st.set_page_config(page_title="CLT Playground", layout="centered")

st.title("Central Limit Theorem (CLT) Interactive Playground")
st.write(
    "Pick a distribution, then watch how the distribution of the sample mean becomes approximately normal as sample size grows."
)

dist = st.selectbox("Distribution", ["Exponential", "Uniform", "Bernoulli"])
n = st.slider("Sample size (n)", min_value=1, max_value=500, value=30, step=1)
m = st.slider("Number of repeated samples", min_value=200, max_value=10000, value=2000, step=200)

rng = np.random.default_rng(42)

def get_sample_means(dist_name: str, m: int, n: int):
    if dist_name == "Exponential":
        x = rng.exponential(scale=1.0, size=(m, n))
        mu, var = 1.0, 1.0
    elif dist_name == "Uniform":
        x = rng.random(size=(m, n))
        mu, var = 0.5, 1.0 / 12.0
    else:
        p = st.slider("Bernoulli p", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
        x = rng.binomial(n=1, p=p, size=(m, n))
        mu, var = p, p * (1 - p)

    return x.mean(axis=1), mu, var

means, mu, var = get_sample_means(dist, m, n)
sigma_mean = sqrt(var / n)

st.subheader("Histogram of sample means")
fig = plt.figure(figsize=(7.5, 4.5), dpi=150)
plt.hist(means, bins=40, density=True)

xs = np.linspace(means.min(), means.max(), 400)
plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma_mean), linewidth=2)

plt.xlabel("Sample mean")
plt.ylabel("Density")
plt.title(f"{dist}: {m} samples of size n={n}")
st.pyplot(fig)

st.subheader("Quick stats")
st.write({
    "Theoretical mean of sample mean": float(mu),
    "Empirical mean of sample mean": float(means.mean()),
    "Theoretical std of sample mean": float(sigma_mean),
    "Empirical std of sample mean": float(means.std(ddof=1)),
})

st.caption("Tip: Increase n to see the histogram become closer to the normal curve.")
