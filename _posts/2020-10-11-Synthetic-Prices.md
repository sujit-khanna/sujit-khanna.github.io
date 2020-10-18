---
layout: post
title: Synthetic Prices for strategy backtesting and parameter estimation
---

``` Note: The ideas and the backtesting methodologies detailed in this article are for illustrative purposes only, it's is in no way the best representation of how strategies must be backtested (same goes for metrics used to judge the strategy performance). The main purpose of this article is to exhibit how synthetic prices can augment/enhance the parameter selection process for a given strategy to reduce overfitting. This``` [GitHub](https://github.com/sujit-khanna/synthetic_prices_using_MCMC) ```repository contains all the notebooks used in this article. ```

This article details, how one can generate and use synthetic prices to avoid false-positive strategies/strategy parameters, which is very common when backtests are performed on actual historical prices. Equity prices generally tend to follow a random path generated from some stochastic process, and actual price is just one such realization of these paths. This makes overfitted parameters hard to identify when strategy backtests are performed on actual prices. Generally testing a strategy on a set of synthetically generated prices can prevent one from overfitting strategy parameters to a single price path.

The synthetic prices are generally extracted via some Data Generating Process or DGP. There are several methods to create/train a DGP, but the most common ones are Generative Adversarial Networks, Autoencoders (Varational and Regular), and Monte Carlo Methods. In this blog, we create a DGP via the Markov Chain Monte Carlo method using NUTS (or No U-Turn Sampler) and see how synthetic prices can improve out-of-sample strategy performance. The DGP in this article is extracted from the stochastic volatility(time-varying) model fit on observed price returns using NUTS. The model and some parts of the code are extracted from the PYMC3 tutorial on stochastic volatility[1], if one is familiar with this model, proceed to the Posterior Predictive Checks [section](###generating-synthetic-prices-from-the-posterior) 

The first step involves defining our Bayesian model, including the priors and generative process for the stochastic volatility model. This model is adapted from the one mentioned in the original paper on NUTS and [1], this model has several distribution for the priors such as exponential distribution for <!-- $\nu$ --> <img style="transform: translateY(0.25em);" src="../svg/ugoBluQfTa.svg"/> and <!-- $\sigma$ --> <img style="transform: translateY(0.25em);" src="../svg/gpiPAzINaV.svg"/> (step size), gaussian random walk <!-- $\mathcal{N}$ --> <img style="transform: translateY(0.25em);" src="../svg/OnDwaYF0q7.svg"/> for the latent volatilities (stochastic) prior. The posterior distribution of returns are modeled using T-distribution <!-- $t$ --> <img style="transform: translateY(0.25em);" src="../svg/DNUlVb8IYt.svg"/>. The model is formally defined as


<!-- $\sigma \sim exp(a)$ --> <p align="center"><img  style="transform: translateY(0.25em);" src="../svg/F1w77g6nis.svg"/></p>
<!-- $\nu \sim exp(b)$ --> <p align="center"><img style="transform: translateY(0.25em);" src="../svg/7K86ZgtUlH.svg"/></p>
<!-- $s_{i} \sim \mathcal{N( s_{i-1}, \sigma^{-2})}$ --> <p align="center"><img style="transform: translateY(0.25em);" src="../svg/taaq159L8J.svg"/></p>

<!-- $log(y_{i}) \sim t(\nu, 0, exp(-2s_{i}))$ --><p align="center"> <img style="transform: translateY(0.25em);" src="../svg/ATfaCOQF8Q.svg"/></p>

Graphically this model is represented as, 


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93732430-4c7f3b80-fb9f-11ea-8a80-e36d7fb4a89c.png" height="200"></p>

With our model formally defined, we proceed with fitting it on the logarithmic returns of the SPY ETF. For generating synthetic returns using the posterior distribution of this model, we use historical log returns for the date range 2013-2018. The data beyond 2018 is considered as the true out of sample data when we compare the backtesting methodologies using actual prices and synthetic prices. The code snippet below loads the SPY prices and defines the models with the priors.

### Loading data and defining model:

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
    spy_df['log_ret'] = (np.log(spy_df.Adj_Close) - np.log(spy_df.Adj_Close.shift(1))).dropna()


    def make_stochastic_volatility_model(data):
        with pm.Model() as model:
            step_size = pm.Exponential('step_size', 10)
            volatility = pm.GaussianRandomWalk('volatility', sigma=step_size, shape=len(data))
            
            nu = pm.Exponential('nu', 0.001)
            returns = pm.StudentT('returns',
                            nu=nu,
                            lam=np.exp(-2*volatility),
                            observed=data["log_ret"])
        return model

    stochastic_vol_model = make_stochastic_volatility_model(spy_df)

It is generally a good practice to visualize the samples from the priors and compare it to the actual data. For faster convergence, we must choose appropriate priors and their corresponding parameters. Misspecifying the model can either lead to delayed or no convergence. Looking at the visuals below, we can observe that priors do not seem to be initialized at an unrealistic starting point (neither too restrictive nor too loose)

### Sampling and Visualizing the Priors:

    with stochastic_vol_model:
        prior = pm.sample_prior_predictive(500)
    fig, ax = plt.subplots(figsize=(14, 4))
    spy_df['log_ret'].reset_index().plot(ax=ax, lw=1, color='black')
    ax.plot(prior['returns'][4:6].T, 'g', alpha=0.5, lw=1, zorder=-10)

    max_observed, max_simulated = np.max(np.abs(spy_df['log_ret'])), np.max(np.abs(prior['returns']))
    ax.set_title(f"Maximum observed: {max_observed:.2g}\nMaximum simulated: {max_simulated:.2g}(!)");

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93733376-1fcd2300-fba3-11ea-8b70-13a8bb60d4a9.png" height="200"></p>

We now fit the model using the NUTS sampler on the observed log returns. We use burned trace, as the first half samples generated by the posterior are generally suboptimal, hence it's a good practice to burn/avoid them in the final analysis. With our model trained, let's plot its trace to visualize the posterior samples (volatility and returns).

### Fitting the Model on Observed Data using NUTS and Visualizing the Posteriors:

    with stochastic_vol_model:
        trace = pm.sample(2000, tune=8000, target_accept=0.9, start=start)
        burned_trace = trace[1000:]

    fig, axes = plt.subplots(nrows=2, figsize=(14, 7))
    spy_df['log_ret'].plot(ax=axes[0], color='black')
    axes[1].plot(x_vals, np.exp(burned_trace['volatility'][::5].T), 'r', alpha=0.5, )
    axes[0].plot(x_vals, posterior_predictive['returns'][::5].T, 'g', alpha=0.5, zorder=-10)
    axes[0].set_title("True log returns (black) and posterior predictive log returns (green)")
    axes[1].set_title("Posterior volatility")


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93793078-58511900-fc04-11ea-9f27-dc0b094ccfbd.png"></p>

### Generating Synthetic Prices from the Posterior

Once we fit the posterior to the observed data using NUTS, we can then use it to generate data sets using the parameter settings of the samples drawn from the posterior. This is done using PPC or Posterior Predictive Checks. According to Rubin(1984), PPC can be summarized as 


```Given observed data X_obs, what would we expect to see in hypothetical replications of the```
```study that generated X_obs? Intuitively, if the model specifications are appropriate we```
```would expect to see something similar to what we saw originally, at least similar in relevant ways. ```

In other words,
**PPC analyzes the degree to which data generated from the model deviates from the data generated from the true distribution**
Once we perform PPC by drawing samples from the posterior, we can perform a quick check if the model is generating patterns similar to the observed data, by visualizing the distribution of the samples and the observed returns.

```
    with stochastic_vol_model:
        posterior_predictive = pm.sample_posterior_predictive(burned_trace)
    gen_data = az.from_pymc3(burned_trace, posterior_predictive=posterior_predictive)
    az.plot_ppc(gen_data)
```

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93793906-6eaba480-fc05-11ea-86c6-08aa642aa320.png"></p>

Looking at the above plot, the model closely approximates the true distribution, however, the true distribution still seems to have fatter tails compared to the sample mean distribution. The next code block reconstructs the SPY ETF prices from the observed return distribution and sample distributions.

```
# Plot original returns 
act_idx = spy_df["Adj_Close"].iloc[0]*(np.exp(spy_df['log_ret'])).cumprod()
act_idx.plot(figsize=(10, 6), title= "True SPY Price")
```
<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93795165-0231a500-fc07-11ea-9457-8bb7bae547ca.png"></p>

```
# Plot synthetic returns 
from pylab import rcParams
fig = plt.figure()
rcParams['figure.figsize'] = 10, 6
for i in range(len(posterior_predictive['returns'])):
    plt.plot(spy_df["Adj_Close"].iloc[0]*(np.exp(posterior_predictive['returns'][i])).cumprod())
fig.suptitle('Synthetic SPY Prices')
```

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93795337-3d33d880-fc07-11ea-84ed-b5fe9295dd63.png"></p>

The PPC generates about 4000 different samples and looking at the plots above. We observe that certain prices, do not follow a similar trajectory to the observed prices (i.e. a few synthetic prices have negative mean). We can further eliminate unwanted prices, by selecting the synthetic returns that closely resemble the actual returns distribution. This can usually be done using distribution based distance measures like **Wasserstein distances, Kullback–Leibler divergence, Jensen–Shannon divergence etc**. In this article we select the top <!-- $n$ --> <img style="transform: translateY(0.25em);" src="../svg/ojecHSYwnA.svg"/> synthetic samples that have the least Wasserstein-1 distance to the observed returns. 

The code block below selects the top <!-- $n$ --> <img style="transform: translateY(0.25em);" src="../svg/trTddHiKdb.svg"/>, sample returns with the lowest wasserstein distance to the observed returns.

```
synthetic_df = pd.DataFrame(posterior_predictive['returns']).T
synthetic_df.columns = [f"price {str(x)}" for x in range(posterior_predictive['returns'].shape[0])]
actual_df = spy_df['log_ret'].to_frame()

from scipy.stats import wasserstein_distance
min_threshold=50 # number of synthetic samples
price_list = []
for name, cols in synthetic_df.items():
    cols = pd.to_numeric(cols, errors='coerce').ffill()
    kl_d = wasserstein_distance(actual_df.values.flatten(), cols.values)
    df = pd.DataFrame({"series_name": name, "kl_d": kl_d}, index=[1])
    price_list.append(df)
full_df = pd.concat(price_list)
full_df = full_df.sort_values(["kl_d"], ascending=True)
best_prices = full_df["series_name"].iloc[:min_threshold].values.tolist()
selected_df = synthetic_df[best_prices]
selected_df.to_csv(f'./SYN_SPY_2018_top{min_threshold}')
```

The plot below, compares the top 50 synthetic prices with the lowest wasserstein-1 distance (on returns) to observed prices.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/93799199-d2859b80-fc0c-11ea-9021-42721de9cc22.png"></p>


### Backtesting with Actual and Synthetic Prices:
Now that we have a DGP, that seems to be producing relevant prices, the next step involves creating a Toy strategy and backtesting it on actual and synthetic prices to assess the effectiveness of the two methods. The toy strategy used in this analysis is called <!-- $Bollinger \; Breakout \; Strategy$ --> <img style="transform: translateY(0.25em);" src="../svg/25OAtBykTs.svg"/>. Which is a variation on a regular bollinger strategy, but goes long/short when it crosses the upper and lower bands, and squares off when the prices cross the mid-band. The code block below shows the function that generates signals for this strategy.

```
def gen_bband_signals(df, lbk, band_dev):
    close_price = df.values
    u_band, l_band, m_band = ta.BBANDS(close_price, timeperiod=lbk, nbdevup=band_dev,
                                           nbdevdn=band_dev, matype=0)
    
    bb_signals = np.asarray(np.zeros(close_price.shape)).astype(float)
    for i in range(lbk, len(bb_signals) - 1):
        if close_price[i] > u_band[i]:
            bb_signals[i] = 1
        elif close_price[i] < u_band[i] and close_price[i] >= m_band[i] and bb_signals[i - 1] == 1:
            bb_signals[i] = 1
        elif close_price[i] < l_band[i]:
            bb_signals[i] = -1
        elif close_price[i] > l_band[i] and close_price[i] <= m_band[i] and bb_signals[i - 1] == -1:
            bb_signals[i] = -1
        else:
            bb_signals[i] = 0
    
    return pd.Series(bb_signals, index=df.index)
```

Looking at the above function, we see that the strategy primarily uses 2 parameters, the lookback period i.e. (lbk), and deviation multiplier (band_dev). In this section, we will analyze a host of parameter sets on in-sample data (till 2018) using both synthetic prices and actual prices. Then compare the performance of these parameters on the observed out-of-sample data (2019-2020). The code block below runs all possible combination of parameters *lbk* and *band_dev* on synthetic prices and the actual price series, where *param_list1=[10, 20, 40, 60, 120]* corresponds to *lbk* parameter and *param_list2=[1, 1.5, 2, 2.5, 3]* corresponds to *band_dev* parameter.  

``` Note: For sake of simplicity transaction costs and slippages are not considered in these backtests.```

```
import itertools
def param_gen(param_list1, param_list2):
    return [params for params in itertools.product(param_list1, param_list2)]

def run_param_simulation(prices_df, param_list1, param_list2):
    param_sets = param_gen(param_list1, param_list2)
    mean_ret_list, stdev_list, param_list = [], [], []
    pctile_list = []
    for params in param_sets:
        strat_signals = prices_df.apply(gen_bband_signals, args = [params[0], params[1]])
        strat_returns = prices_df.pct_change(1)
        strat_perf = strat_signals.shift(2)*strat_returns
        mean_ret_list.append(strat_perf.sum(axis=0).mean())
        stdev_list.append(strat_perf.sum(axis=0).std())
        param_list.append(f"lbk={params[0]}_band={params[1]}")
        pctile_list.append(np.percentile(strat_perf.sum(axis=0), 0.1))
        
    return pd.DataFrame({"params": param_list, "mean_return":mean_ret_list, "stdev_return":stdev_list, "10_percentile_return": pctile_list})
    
    # running all parameters on synthetic prices
    syn_params_df = run_param_simulation(syn_prices_df, [10, 20, 40, 60, 125], [1, 1.5, 2, 2.5, 3])

    # running all parameters on actual prices
    act_trunc_params_df = run_param_simulation(act_df_trunc["Adj_Close"].to_frame(), [10, 20, 40, 60, 125], [1, 1.5, 2, 2.5, 3])
```

To analyze the performance of each strategy run (i.e. an independent parameter set) on synthetic prices we created 3 metrics, **mean strategy return**, **he standard deviation of strategy return (i.e. across different synthetic prices)**, and **10th percentile return of the strategy**.
The top 10 parameter sets on the in-sample period (2013-2018) sorted by mean strategy returns, for synthetic and actual prices are shown in the table below.

       
```
syn_params_df.sort_values(by=["mean_return"], ascending=False).head(10)

=======================================================================
idx         params	 mean_return	stdev_return 10_percentile_return
13	lbk=40_band=2.5	  0.115696	  0.299428	 -0.377473
9	lbk=20_band=3	  0.105347	  0.322414	 -0.402350
14	lbk=40_band=3	  0.131550	  0.314481	 -0.407950
3	lbk=10_band=2.5	  0.054177	  0.282541	 -0.422210
1	lbk=10_band=1.5	  0.115554	  0.278454	 -0.432102
24	lbk=125_band=3	  0.063512	  0.291127	 -0.433385
21	lbk=125_band=1.5  0.102877	  0.294643	 -0.446396
22	lbk=125_band=2	  0.128665	  0.313494	 -0.473440
2	lbk=10_band=2	  0.137536	  0.241867	 -0.524847
17	lbk=60_band=2	  0.142380	  0.342193	 -0.548139
```

The *stdev_return* is *NaN* and *10_percentile_return* is same as *mean_return* for actual prices, as there is only one price path for in the case of actual/observed prices.
```
act_trunc_params_df.sort_values(by=["mean_return"], ascending=False).head(10)

=======================================================================
idx         params	  mean_return stdev_return 10_percentile_return
21	lbk=125_band=1.5   0.352037	   NaN	         0.352037
20	lbk=125_band=1	   0.340561	   NaN	         0.340561
22	lbk=125_band=2	   0.295713	   NaN	         0.295713
15	lbk=60_band=1	   0.269968	   NaN	         0.269968
23	lbk=125_band=2.5   0.124201	   NaN	         0.124201
16	lbk=60_band=1.5	   0.030204	   NaN	         0.030204
13	lbk=40_band=2.5	   0.020785	   NaN	         0.020785
17	lbk=60_band=2	  -0.022522	   NaN	        -0.022522
11	lbk=40_band=1.5	  -0.038694	   NaN	        -0.038694
12	lbk=40_band=2	  -0.104843	   NaN	        -0.104843
```

Now that we have some performance measure for the strategy parameters, we can proceed with creating a framework that selects the best possible parameter based on its performance on synthetic prices.  


Note: 
* Since we only have a single path for the actual prices, we will compare the best performing parameter in this space (by mean_returns or returns), with the parameters identified by our framework on synthetic prices.
* We are only considering strategy return based metrics and avoiding other risk and risk-adjusted metrics, as we do not want to increase the dimensionality of the problem in this Toy example.
* The strategy chosen for this example is only for illustration, for all practical purposes this strategy would have been eliminated from the analysis by looking at really poor *10_percentile_return* metric values.

We define a basic framework to select the best possible parameter set using synthetic prices where we take the ***intersection of the best possible strategy parameters across all three metrics***. The key idea here is to implement a very restrictive filter where the strategy parameter is chosen only when it performs well across all three metrics. The tables below show the Out-Of-Sample performance (2019-2020), for the best performing parameter sets on actual and synthetic prices.

### Running Out-of-Sample performance on best parameter found on actual in-sample prices

```
import re
param_list = re.findall(r'''(\d+(?:\.\d+)*)''', act_trunc_params_df.sort_values(by=["mean_return"], ascending=False)["params"].iloc[0])
best_act_param_df = run_param_simulation(act_df_os["Adj_Close"].to_frame(), [int(param_list[0])], [float(param_list[1])])
best_act_param_df

======================================================================================
idx	     params	     mean_return stdev_return 10_percentile_return
0	lbk=125_band=1.5  -0.067634	     NaN	     -0.067634
```

### Out-of-Sample performance of parameters generated from synthetic prices (highest mean_return)
  
```
param_list = re.findall(r'''(\d+(?:\.\d+)*)''', syn_params_df.sort_values(by=["mean_return"], ascending=False)["params"].iloc[0])
best_mean_param_df = run_param_simulation(act_df_os["Adj_Close"].to_frame(), [int(param_list[0])], [float(param_list[1])])
best_mean_param_df

======================================================================================
idx          params	 mean_return    stdev_return 10_percentile_return
0	lbk=40_band=2.0	   0.049309	    NaN	         0.049309
```

### Out-of-Sample performance of parameters generated from synthetic prices (lowest standard deviation with positive mean_return)

```
param_list = re.findall(r'''(\d+(?:\.\d+)*)''', syn_params_df.loc[syn_params_df["mean_return"]>=0].sort_values(by=["stdev_return"], ascending=True)["params"].iloc[0])
best_std_param_df = run_param_simulation(act_df_os["Adj_Close"].to_frame(), [int(param_list[0])], [float(param_list[1])])
best_std_param_df

======================================================================================
idx	     params	mean_return stdev_return 10_percentile_return
0	lbk=10_band=2.0	  0.085758	NaN	     0.085758
```

### Out-of-Sample performance of parameters generated from synthetic prices (highest 10 percentile return, with positive mean_return)

```
param_list = re.findall(r'''(\d+(?:\.\d+)*)''', syn_params_df.loc[syn_params_df["mean_return"]>=0].sort_values(by=["10_percentile_return"], ascending=False)["params"].iloc[0])

best_pct_param_df = run_param_simulation(act_df_os["Adj_Close"].to_frame(), [int(param_list[0])], [float(param_list[1])])
best_pct_param_df

======================================================================================
idx          params	  mean_return stdev_return 10_percentile_return
0	lbk=40_band=2.5	   0.103046	  NaN	        0.103046
```

### Out-of-Sample performance of parameters generated from synthetic prices on best intersected parameters from mean_return, standard deviation and 10 percentile results

```
best_returns = syn_params_df.sort_values(by=["mean_return"], ascending=False).head(10)["params"]
best_stdev = syn_params_df.loc[syn_params_df["mean_return"]>=0].sort_values(by=["stdev_return"], ascending=True).head(10)["params"]
best_10_pct = syn_params_df.loc[syn_params_df["mean_return"]>=0].sort_values(by=["10_percentile_return"], ascending=False).head(10)["params"]
# best_params = best_returns.intersection(best_stdev).intersection(best_10_pct)
best_params = list(set(best_returns).intersection(set(best_stdev)).intersection(set(best_10_pct)))
best_params_perf = syn_params_df.loc[syn_params_df["params"].isin(best_params)]  
best_params_perf # in-sample synthetic price performance

======================================================================================

idx	    params    mean_return   stdev_return  10_percentile_return
1	lbk=10_band=2	0.137536       0.241867	       -0.524847

```


```
best_act_param_df = run_param_simulation(act_df_os["Adj_Close"].to_frame(), [10], [2])
best_act_param_df # out-of-sample actual price performance

======================================================================================

idx         params    mean_return stdev_return 10_percentile_return
0	lbk=10_band=2   0.085758      NaN	    0.085758

```

Based on the tables above, we observed that the best parameter set chosen based on in-sample actual prices tends to vastly underperform in the out-of-sample [periods](###running-out-of-sample-performance-on-best-parameter-found-on-actual-in-sample-prices). Conversely, the parameter set chosen based the framework defined for synthetic prices had the out-of-sample performance profile similar to the in-sample synthetic prices [profile](###out-of-sample-performance-of-parameters-generated-from-synthetic-prices-on-best-intersected-parameters-from-mean_return,-standard-deviation-and-10-percentile-results)

### Conculsion and Next Steps:
The article above details how one can use synthetic prices generated by the posterior distribution of our Bayesian model to minimize the risk of finding false positive strategy parameters or in other words reduce the risk of overfitting (prevalent when actual prices are used). The parameter selection framework described in this article only looks at total cumulative returns as a metric to identify strategy parameters, however, one can use risk-adjusted or more sophisticated performance measures in this framework. Another advantage of using synthetic prices is that we can perform statistical analysis of performance measures, and perform distribution analysis to come up with additional insights about the performance of the strategy..


Additional Notes:
*   Similar to the univariate time series model described, multi-variate synthetic prices can also be generated using Cholesky decomposition.
*   Synthetic Prices can also be generated via Time-Series GAN models or Varational Autoencoders. However, GAN models are highly susceptible to mode-collapse and require appropriate tuning of hyper-parameters. Based on my experience both GANs and VAEs tend to create synthetic prices very similar to the actual prices, i.e. the prices generated by these models tend to exhibit less variation with respect to observed/actual prices.
*   We can very conveniently create a tactical trading model that optimizes a strategy based on the outcome of a certain scenario. As trading strategies tend to exhibit different behavior under different market regimes, creating synthetic prices for different regimes can help in mapping strategy's performance to market regimes. This information can then be used tactically to switch ON/OFF or change the allocation of a particular strategy
*   The next article defines a methodology for creating such a tactical trading framework.

The repo and notebook can be accessed here [synthetic price generation](https://github.com/sujit-khanna/synthetic_prices_using_MCMC/blob/main/mcmc_synthetic_prices.ipynb) and [parameter selection](https://github.com/sujit-khanna/synthetic_prices_using_MCMC/blob/main/param_selection_using_MCMC_prices.ipynb)

## References ##
    1. Stochastic Volatility Model, https://docs.pymc.io/notebooks/stochastic_volatility.html
    2. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo,
       https://arxiv.org/pdf/1111.4246.pdf
    3. Probabilistic Programmming in Python using PYMC3, https://arxiv.org/pdf/1507.08050.pdf

