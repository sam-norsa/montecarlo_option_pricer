import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

#Price is the expected discounted pay-off
#We assume the spot price follows a log-normal distribution

def calc_euro_option_price_under_rn(mu, sigma, strike, maturity, discount_rate, sim_size):
    S=np.exp(mu + sigma*rd.randn(sim_size))
    returns = np.maximum(S-strike,0)
    expected_payoff = (discount_rate**maturity)*np.mean(returns)
    return expected_payoff

print(f"The estimated price is {calc_euro_option_price_under_rn(1,0.1,1,10,0.95,100000):3f}")
