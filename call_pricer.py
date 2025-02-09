import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Price is the expected discounted pay-off
# We assume the spot price follows a log-normal distribution

def calc_euro_option_price_under_rn(mu, sigma, strike, maturity, discount_rate, sim_size):

    S = np.exp(mu + sigma*rd.randn(sim_size))
    returns = np.maximum(S-strike, 0)
    expected_payoff = (discount_rate**maturity)*np.mean(returns)
    return expected_payoff

""" Let's implement a more realistic version, considering log(S_t+1/S_t) = mu + sigma_t*X_t+1
where sigma_t = exp(h_t) and h_t+1 = a*h_t + b*eta_t+1 and eta_t is iid N(0,1) """

#The latter yields log(S_t+1) = log(S_t) + mu + sigma_t*X_t+1

def sim_asset_price_path(S_0, mu, h, maturity, a, b):
    prices = np.empty(maturity+1)
    prices[0] = np.log(S_0)
    for k in range(1, maturity+1):
        prices[k] = prices[k-1]+mu+np.exp(h)*rd.randn()
        h = a*h + b*rd.randn()
    return np.exp(prices)

S_0 = 1
mu = 0.01
h = 0.1
maturity = 100
a = 0.8
b = 0.1
num_paths = 100

plt.figure(figsize=(10, 5))
for _ in range(num_paths):
    prices = sim_asset_price_path(S_0, mu, h, maturity, a, b)
    plt.plot(range(maturity + 1), np.log(prices), linewidth=0.8, alpha=0.7)

plt.title("Log Paths")
plt.xlabel("Temps")
plt.ylabel("Log(Prix)")
plt.grid(True)
plt.savefig('sim.png')