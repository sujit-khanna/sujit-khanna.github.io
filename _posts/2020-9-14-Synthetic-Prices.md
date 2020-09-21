---
layout: post
title: Synthetic Prices for strategy backtesting and tactical trading
---

This article details, how one can generate and use synthetic prices to avoid false positive strategies, which is very commong when backtests are performed on actual historical prices. Equity prices generally tend to follow a random path generated from some stochastic process, and actual price is just one such realization of these paths. This makes overfitting parameters hard to identify, when backetsting the strategy on actual prices. Generally testing a strategy on a set of synthetically generated prices prevents one from overfitting strategy parameters to a single price path. 

The synthetic prices are generally extracted via some Data Generating Process or DGP. There are several methods to create/train a DGP, but the most common ones are Generative Advesarial Networks, Autoenconders (Varational and Regular), and Monte Carlo Methods. In this blog we create a DGP via Markov Chain Monte Carlo method using NUTS (or No U-Turn Sampler), and see how synthetic prices can improve out-of-sample strategy performance. The DGP in this article is is extracted from a stochastic volatility(time-varying) model fit on observed price returns using NUTS.

The first step involves defining our bayesian model, including the priors and generative process for the stochastic volatility model. This model is adapted from the one mentioned in orignial paper on NUTS and [1], this model has several distribution for the priors such as exponential distribution for <!-- $\nu$ --> <img style="transform: translateY(0.25em);" src="../svg/ugoBluQfTa.svg"/> and <!-- $\sigma$ --> <img style="transform: translateY(0.25em);" src="../svg/gpiPAzINaV.svg"/> (step size), gaussian random walk <!-- $\mathcal{N}$ --> <img style="transform: translateY(0.25em);" src="../svg/OnDwaYF0q7.svg"/> for the latent volatilities (stochastic) prior. The posterior distribution of returns are modeled using T-distribution <!-- $t$ --> <img style="transform: translateY(0.25em);" src="../svg/DNUlVb8IYt.svg"/>. The model is formally defined as <br/>

<!-- $\sigma \sim exp(a)$ --> <p align="center"><img  style="transform: translateY(0.25em);" src="../svg/F1w77g6nis.svg"/></p>
<!-- $\nu \sim exp(b)$ --> <p align="center"><img style="transform: translateY(0.25em);" src="../svg/7K86ZgtUlH.svg"/></p>
<!-- $s_{i} \sim \mathcal{N( s_{i-1}, \sigma^{-2})}$ --> <p align="center"><img style="transform: translateY(0.25em);" src="../svg/taaq159L8J.svg"/></p>

<!-- $log(y_{i}) \sim t(\nu, 0, exp(-2s_{i}))$ --><p align="center"> <img style="transform: translateY(0.25em);" src="../svg/ATfaCOQF8Q.svg"/></p>
s
Graphically this model is represented as, </br>

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93732430-4c7f3b80-fb9f-11ea-8a80-e36d7fb4a89c.png" height="200"></p>

With our model formally defined, we proceeed with fitting it on log returns of the SPY ETF, for generating synthetic returns using the posteror distribution of this model, we use historical log returns for the date range 2013-2018. The data beyond 2018 is considered as the true out of sample data when we compare the backtesting methodologies using actual prices and synthetic prices.
The code snippet below read the SPY prices, and defines the models with the priors.

***Loading data and defining model:***

    import matplotlib.pyplot as plt
    import arviz as az

    import numpy as np
    import pandas as pd
    import pymc3 as pm
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    np.random.seed(0)
    az.style.use("arviz-darkgrid")

    filepath = f"/Users/sujitkhanna/Desktop/Talos/marketing_material/misc_material/tactical trading framework/SPY_2018.csv"
    spy_df = pd.read_csv(filepath)
    spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.date
    spy_df = spy_df.set_index("date")
    # spy_df['log_ret'] = (np.log(spy_df.Adj_Close) - np.log(spy_df.Adj_Close.shift(1))).dropna()


    def make_stochastic_volatility_model(data):
    with pm.Model() as model:
        step_size = pm.Exponential('step_size', 10)
        volatility = pm.GaussianRandomWalk('volatility', sigma=step_size, shape=len(data))
        
        nu = pm.Exponential('nu', 0.001)
        returns = pm.StudentT('returns',
                        nu=nu,
                        lam=np.exp(-2*volatility),
                        observed=data["returns"])
    return model

    stochastic_vol_model = make_stochastic_volatility_model(spy_df)

It is generally a good practice to visualize the samples from the priors and compare it to the actual data. For faster convergence, it's necessary that we choose appropriate priors and their corresponding parameters. Misspecifying the model can either lead to delayed or no convergence. Looking at the visuals below, we can observe that priors do not seem to be initialized at a unrealistic starting point (neither too restrictive nor too loose).

***Sampling and Visualizing the Priors:***

    with stochastic_vol_model:
        prior = pm.sample_prior_predictive(500)
    fig, ax = plt.subplots(figsize=(14, 4))
    spy_df['log_ret'].reset_index().plot(ax=ax, lw=1, color='black')
    ax.plot(prior['returns'][4:6].T, 'g', alpha=0.5, lw=1, zorder=-10)

    max_observed, max_simulated = np.max(np.abs(spy_df['log_ret'])), np.max(np.abs(prior['returns']))
    ax.set_title(f"Maximum observed: {max_observed:.2g}\nMaximum simulated: {max_simulated:.2g}(!)");

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93733376-1fcd2300-fba3-11ea-8b70-13a8bb60d4a9.png" height="200"></p>


We now fit the model using NUTS sampler on the observed log returns, we use burned trace, as the first half samples generated by the posterior are generally suboptimal, hence its a good practice to burn/avoid them in the final analysis. With our model trained, lets plot its trace to visualize the the posterior samples (volatility and returns).

***Fitting the Model on Observed Data using NUTS and Visualizing the Trace:***

    with stochastic_vol_model:
        trace = pm.sample(2000, tune=8000, target_accept=0.9, start=start)
        burned_trace = trace[4000:]










## References ##
    1. Stochastic Volatility Model, https://docs.pymc.io/notebooks/stochastic_volatility.html
    2. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo,
       https://arxiv.org/pdf/1111.4246.pdf
    3. Probabilistic Programmming in Python using PYMC3, https://arxiv.org/pdf/1507.08050.pdf

