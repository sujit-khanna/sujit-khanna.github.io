---
layout: post
title: Tactical Trading Using Synthetic Prices 
---
This is a continuation of the previous article on generating synthetic prices for strategy parameter selection (add link). In this article we discuss how one can augment the framework defined in the previous [article](https://sujit-khanna.github.io/Synthetic-Prices/) to generate specific scenario based prices. ***Tactical Investment Algorithms*** Paper by Marcos Lopez de Prado [1](###references), is one of the best papers around that details the power of using synthetic prices and MC backtests to find trading strategies optimized to work in a particular market regime i.e. trading tactically. </br>

 It is generally difficult to create strategies that perform consistently across different market regimes, for example a simple strategy that shorts VIX futures performs well during low volatility regimes, but rakes in a lot of losses during high volatility periods. The [volmageddon](https://www.zerohedge.com/news/2018-02-06/first-volmageddon-casualties-emerge-one-hedge-fund-down-much-65) event of 2018 and covid crisis 2020 are two prime examples where such a strategy would not only wipe out all the previous profits but also incur tremendous losses in a short period. Creating synthetic prices that reflect a particular regime can be a very effective tool in mapping a strategies performance in different market regimes, this relationship (if significant) can be effectively used to switch on/off strategies or change allocations of strategies to maximize profitability (raw or risk adjusted).

The article below describes two methods via which one can generate synthetic prices for different market scenarios/regimes, and later shows how we can use different scenarios to dynamically adjust the strategy parameters such that it works well across different market regimes. The market scenarios defined in this articles reflect the market regimes. 
More specifically we see markets operating in 2 simple regimes low volatility regime and high volatility regime. 

### Regime Switching Model
The first method involves defining our model as a mixture of two student T random walks, that switches based on the likelihood of operating within these two distributions. The two student T distributions represents the 2 market regimes/states we're trying to model (high/low volatility), and the likelyhood of being in a particular state is defined using markov/stochastic matrix which contains the transistion probabilities from one state to another (see below). </br>

<!-- $ p_{state_i} = \begin{pmatrix}
p_{(state1->state1)} & p_{(state1->state2)}\\
p_{(state2->state1)} & p_{(state2->state2)}
\end{pmatrix} $ --> <img style="transform: translateY(0.25em);" src="../svg/pBlfAwTHVP.svg"/>

</br> Graphically this model can be represented as </br>

<img src="https://user-images.githubusercontent.com/71300644/94458366-d886f980-0183-11eb-8b36-67b76b5d45b6.png">

The posterior of this model will contain the two distributions corresponding to different regimes,as well as the likelihood of being in a particular regime. With this posterior we can then draw samples, select a particular threshold of switching probability and reconstruct composite synthetic prices using the two random walks (corresponding to 2 regimes). Once we generate the composite prices along with the sections of the price that belong to a particular regime, strategies can be backtested and it's performance can be mapped to different regimes that corresponding to 2 states based on the threshold of the switching probabilities. 

One of the biggest issues with the method described above is that the fitting process is computationally very expensive. Such poor convergence properties are mostly down to an exponentialy large number of parameters that needs to be fit using our sampler. A quick work around to this method is described below, which we will use in our toy example.

### Independent Regime Models
The second method is a little crude and involves splitting the prices/log returns based on differnt market regimes and fitting them independently using the model described in the previous [article](https://sujit-khanna.github.io/Synthetic-Prices/).
One can find a plethora of models that define market regimes few examples include using a markov regime switching model, that oscillates between 2 regimes based on the switching probability or the likelihood of being in a particular state, or one can even use a time-series clustering model to define market clusters that reflect different regimes. 

In this article we use a very simple defination/model to define low and high volatility regimes, which is based on the VIX index. Since the security used in this toy example is the SPY ETF, VIX index seems like a fair reflection of its volatility regime. We say the market regime is currently in low volatility state if the exponential weighted moving average ***(ewm)*** of VIX index is below a certain threshold and in high volatility if it above that threshold (*we can have more regimes but for the sake of simplicity we move ahead with just 2 regimes*). The code block below splits the original price returns into 2 dataframes/csv files based in *ewm* of VIX index and the regime threshold. As shown below we use a ***half_life*** of 10 days and ***VIX Threshold** of 25 to define our regimes. 


<!-- $ if \; VIX \_EWM_{halflife=10days} <= 25 \; , then \; regime = low \_ volatility$ --> <img style="transform: translateY(0.25em);" src="../svg/gYHCXcO8DY.svg"/>
<!-- $ else \; regime = high \_ volatility$ --> <img style="transform: translateY(0.25em);" src="../svg/SglUk5CXTg.svg"/>


```

code block 1 here

```

We will use the notebook from the pervious article to create synthetic prices corresponding to low and high volatility regimes. The figure below plots the reconstructed prices for each regime of the ***SPY ETF***. The plots on the left show the reconstructed price series from true/observed returns, and the ones on the left show the synthetic prices for each regime. The procedure used to select best prices is similar to the one described in the previous article, and 50 best prices with the lowest wasserstein-1 distance are selected in this article as well.

<img src="https://user-images.githubusercontent.com/71300644/94477177-8784fe80-019f-11eb-874d-9a8ef14d3021.png">


<img src="https://user-images.githubusercontent.com/71300644/94477853-6e308200-01a0-11eb-974b-51e4abb74fe4.png">


## Optimizing the strategy parameters based on market regimes
This section deals with how to use the regime based synthetic prices to switch between appropriate parameters based on the market regime. Generally even for an extremely robust strategy a single set of parameters do not tend to perform consistently across all regimes. During periods of high uncertainty, strategies need to operate with parameters that are more sensitive to the changes in the market environment, and during low volatility periods it's generally better to have parameters that are less responsive to avoid getting trapped in small random market movements. The process of switching the strategy parameters can be automated by either making these parameters a function of market prices or volatility, or one can use fixed rules to switch between these parameters. </br>

Using  actual prices is generally not recommended to perform this exercise as we can easily overfit the parameters across different regimes since there is only one price path under consideration which is not true if we have a decent sample size of synthetic prices for each regime. 

### Backtesting across high and low volatility regimes using synthetic prices
Like before we use *Bollinger Breakout Strategy* as our toy strategy in this exercise, and test it's performance on high and low volatility regime synthetic prices. The code block below runs the *Bollinger Breakout Strategy* on high and low volatility regime prices and outputs the total returns generated by each parameter set in both regimes. 

```Note: For sake of simplicity transaction costs and slippages are not considered in these backtests.```

```
def gen_bband_signals(df, lbk, band_dev):
    close_price = df.values
    u_band, l_band, m_band = ta.BBANDS(close_price, timeperiod=lbk, nbdevup=band_dev,
                                           nbdevdn=band_dev, matype=3)
    
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
```

In this article we have used a much broader parameter space to test the strategy parameters as we're testing on two regimes separately and two regimes will impace the parameters differently.

#### Backtest results on Low Volatility Synthetic Prices 
```
regime1_params_df = run_param_simulation(regime_1_price_df, [10, 20, 40, 50, 60, 80, 125], [1, 1.5, 2, 2.5, 3])
regime1_params_df.sort_values(by=["mean_return"], ascending=False).head(10)
===========================================================================

idx	    params	   mean_return stdev_return	10_percentile_return
13	lbk=40_band=2.5	0.115701	0.238962	-0.288328
17	lbk=50_band=2	0.103606	0.265799	-0.405246
7	lbk=20_band=2	0.095941	0.216310	-0.346659
26	lbk=80_band=1.5	0.084921	0.272712	-0.506629
12	lbk=40_band=2	0.084603	0.263745	-0.406878
19	lbk=50_band=3	0.081826	0.277830	-0.346280
25	lbk=80_band=1	0.080644	0.252592	-0.529578
22	lbk=60_band=2	0.080461	0.259887	-0.588156
11	lbk=40_band=1.5	0.079768	0.265743	-0.567365
28	lbk=80_band=2.5	0.076621	0.275044	-0.307758
```

#### Backtest results on High Volatility Synthetic Prices 
The top 10 parameters are chosen by filtering out the parameters when the mean_return is not equal to zero, since the for the cases when *lbk > length of the series*, no trades will be executed. 
```
regime2_params_df = run_param_simulation(regime_2_price_df, [10, 20, 40, 60, 80, 125], [1, 1.5, 2, 2.5, 3])
regime2_params_df[regime2_params_df["mean_return"]!=0].sort_values(by=["mean_return"], ascending=False).head(10)

======================================================================

idx	    params	   mean_return stdev_return 10_percentile_return
1	lbk=10_band=1.5	0.006648	0.093087	-0.213135
5	lbk=20_band=1	-0.000220	0.100214	-0.200437
14	lbk=40_band=3	-0.000949	0.027609	-0.108172
13	lbk=40_band=2.5	-0.001033	0.028484	-0.108172
12	lbk=40_band=2	-0.001415	0.034230	-0.108488
0	lbk=10_band=1	-0.001591	0.108626	-0.158216
6	lbk=20_band=1.5	-0.002021	0.084773	-0.188037
11	lbk=40_band=1.5	-0.002919	0.039170	-0.109136
2	lbk=10_band=2	-0.003158	0.082157	-0.154792
7	lbk=20_band=2	-0.003806	0.070887	-0.151446
```

#### Backtest results on Actual Prices for both regimes
```
# low volatility regime
act1_trunc_params_df = run_param_simulation(spy1_df["recon_Adj_Close"].to_frame(), [10, 20, 40, 60, 80, 125], [1, 1.5, 2, 2.5, 3])
act1_trunc_params_df.sort_values(by=["mean_return"], ascending=False).head(10)
=============================================================================

idx	    params	   mean_return stdev_return   10_percentile_return
1	lbk=10_band=1.5	 0.057940	 NaN	         0.057940
0	lbk=10_band=1	 0.023670	 NaN	         0.023670
26	lbk=125_band=1.5 -0.026388	 NaN	         -0.026388
21	lbk=80_band=1.5	 -0.073611	 NaN	         -0.073611
5	lbk=20_band=1	 -0.076717	 NaN	         -0.076717
2	lbk=10_band=2	 -0.126481	 NaN	         -0.126481
10	lbk=40_band=1	 -0.176319	 NaN	         -0.176319
15	lbk=60_band=1	 -0.182508	 NaN	         -0.182508
16	lbk=60_band=1.5	 -0.200746	 NaN	         -0.200746
11	lbk=40_band=1.5	 -0.204886	 NaN	         -0.204886

```

```
act2_trunc_params_df = run_param_simulation(spy2_df["recon_Adj_Close"].to_frame(), [10, 20, 40, 60, 80, 125], [1, 1.5, 2, 2.5, 3])
act2_trunc_params_df.sort_values(by=["mean_return"], ascending=False).head(10)
==========================================================================
idx   	params	   mean_return stdev_return	10_percentile_return
1	lbk=10_band=1.5	 0.036104	   NaN	         0.036104
10	lbk=40_band=1	 0.012375	   NaN	         0.012375
11	lbk=40_band=1.5	 0.012375	   NaN	         0.012375
12	lbk=40_band=2	 0.012375	   NaN	         0.012375
13	lbk=40_band=2.5	 0.012375	   NaN	         0.012375
14	lbk=40_band=3	 0.012375	   NaN	         0.012375
2	lbk=10_band=2	 -0.012616	   NaN	         -0.012616
3	lbk=10_band=2.5	 -0.012616	   NaN	         -0.012616
4	lbk=10_band=3	 -0.012616	   NaN	         -0.012616
6	lbk=20_band=1.5	 -0.026929	   NaN	         -0.026929

```

### Chosing the best possible parameters from synthetic price and actual price backtests

The parameter selection criteria again is exactly the same as the one we defined in the previous article, where the best parameter based on highest *mean_return* is chosen for actual price backtests, and an intersection based framework is used to select the best possible parameters for each regime.

#### Best parameter set based on backtest results on low volatility synthetic prices
Based on the intersection rule we found 2 parameter candidates for low volatility regime, for the final Out-of-Sample test we choose the parameter set *lbk=40_band=2.5* as it outperforms *lbk=20_band=2* on *mean_return* and *10_percentile_return* metric.

```
best_returns1 = regime1_params_df.sort_values(by=["mean_return"], ascending=False).head(10)["params"]
best_stdev1 = regime1_params_df.loc[regime1_params_df["mean_return"]>0].sort_values(by=["stdev_return"], ascending=True).head(10)["params"]
best_10_pct1 = regime1_params_df.loc[regime1_params_df["mean_return"]>0].sort_values(by=["10_percentile_return"], ascending=False).head(10)["params"]
best_params1 = list(set(best_returns1).intersection(set(best_stdev1)).intersection(set(best_10_pct1)))
best_params_perf1 = regime1_params_df.loc[regime1_params_df["params"].isin(best_params1)]
best_params_perf1
==========================================================================

idx      params    mean_return stdev_return	10_percentile_return
7	lbk=20_band=2	0.095941	0.216310	-0.346659
13	lbk=40_band=2.5	0.115701	0.238962	-0.288328
```

#### Best parameter set based on backtest results on high volatility synthetic prices
On the high volatility regime we just found 1 parameter set that had a positive retusn in the backtests i.e *lbk=10_band=1.5*, even though the return profile seems insignificant it could be down to short length of the high volatility prices. For the sake of completeness we will include this parameter in the final strategy as well. 
``` one can even ignore this parameter and choose not to trade in high volatility regime altogether.```
```
best_returns2 = regime2_params_df.sort_values(by=["mean_return"], ascending=False).head(10)["params"]
best_stdev2 = regime2_params_df.loc[regime2_params_df["mean_return"]>0].sort_values(by=["stdev_return"], ascending=True).head(10)["params"]
best_10_pct2 = regime2_params_df.loc[regime2_params_df["mean_return"]>0].sort_values(by=["10_percentile_return"], ascending=False).head(10)["params"]
best_params2 = list(set(best_returns2).intersection(set(best_stdev2)).intersection(set(best_10_pct2)))
best_params_perf2 = regime2_params_df.loc[regime2_params_df["params"].isin(best_params2)]
best_params_perf2
==========================================================================
idx      params    mean_return stdev_return 10_percentile_return
1	lbk=10_band=1.5	0.006648	0.093087	-0.213135
```

***The best parameter for both high and low volatility actual price (reconstructed) backtests was the same i.e lbk=10_band=1.5***

### Out-of-Sample performance of the dynamic parameter switching strategy
We test the out-of-sample performance on actual prices (2019-2020) using the high and low volatility parameters found using both actual prices and synthetic prices. Since actual (reconstructed) prices has the same parameters for both the regimes we can test it out on the existing strategy. However for using two different strategy parameters based on the volatility regimes, the base strategy would require some modifications.

#### Out-of-Sample performance using parameters found using actual prices

```
best_act_param_df = run_param_simulation(act_df_os["Adj_Close"].to_frame(), [int(param_list[0])], [float(param_list[1])])
best_act_param_df
===========================================================================
	   params	   mean_return	stdev_return	10_percentile_return
0	lbk=10_band=1.5	 0.050112	   NaN	            0.050112

```

#### Out-of-Sample performance using parameters found using synthetic prices
To switch the parameters using the VIX regimes we will need to modify the *bollinger breakout strategy* function to accept 3 more parameters, which are *lbk* and *band_dev* for the second regime and the threshold for defining the *VIX_EMA* regimes. The code block below implements such a function which we will use to test the out-of-sample performance of parameter switching strategy.

```
def gen_bband_multi_regime_signals(df, lbk1, band_dev1, lbk2, band_dev2, regime_thresh):
    close_price = df["Adj_Close"].values
    vix_close_lag1 = df["VIX Close"].shift(1).values
    
    u_band_1, l_band_1, m_band_1 = ta.BBANDS(close_price, timeperiod=lbk1, nbdevup=band_dev1,
                                           nbdevdn=band_dev1, matype=3)
        
    u_band_2, l_band_2, m_band_2 = ta.BBANDS(close_price, timeperiod=lbk2, nbdevup=band_dev2,
                                           nbdevdn=band_dev2, matype=3)
    
    bb_signals = np.asarray(np.zeros(close_price.shape)).astype(float)

    for i in range(max(lbk1, lbk2), len(bb_signals) - 1):

        if vix_close_lag1[i] <= regime_thresh:

            if close_price[i] > u_band_1[i]:
                bb_signals[i] = 1
            elif close_price[i] < u_band_1[i] and close_price[i] >= m_band_1[i] and bb_signals[i - 1] == 1:
                bb_signals[i] = 1
            elif close_price[i] < l_band_1[i]:
                bb_signals[i] = -1
            elif close_price[i] > l_band_1[i] and close_price[i] <= m_band_1[i] and bb_signals[i - 1] == -1:
                bb_signals[i] = -1
            else:
                bb_signals[i] = 0
                
        elif vix_close_lag1[i] > regime_thresh:
            if close_price[i] > u_band_2[i]:
                bb_signals[i] = 1
            elif close_price[i] < u_band_2[i] and close_price[i] >= m_band_2[i] and bb_signals[i - 1] == 1:
                bb_signals[i] = 1
            elif close_price[i] < l_band_2[i]:
                bb_signals[i] = -1
            elif close_price[i] > l_band_2[i] and close_price[i] <= m_band_2[i] and bb_signals[i - 1] == -1:
                bb_signals[i] = -1
            else:
                bb_signals[i] = 0

    
    return pd.Series(bb_signals, index=df.index) 
```

The next step involves augmenting the dataframe containing the actual out-of-sample prices with the VIX_EWM prices. The code block below merges the two prices.

```
vix_file_path = "./vixcurrent.csv"
vix_df = pd.read_csv(vix_file_path)
vix_df["date"] = pd.to_datetime(vix_df["date"])
vix_df = vix_df.set_index("date")
impvol_ema_df = vix_df["VIX Close"].ewm(halflife=10).mean().to_frame()
act_os_vix_df =act_df_os.join(impvol_ema_df, how='left').ffill()
```

Now that we have everything let's run the parameter siwtching strategy and evaluate it's effectiveness. 

```
strat_signals = gen_bband_multi_regime_signals(act_os_vix_df, 40, 2.5, 10, 1.5, 25)
underlying_returns = act_os_vix_df["Adj_Close"].pct_change(1)
strat_perf = strat_signals.shift(2)*underlying_returns
regime_df = pd.DataFrame({"params": ["lbk=40/10_band=2.5/1.5"], "mean_return":strat_perf.sum(axis=0).mean(), "stdev_return":strat_perf.sum(axis=0).std(), "10_percentile_return": np.percentile(strat_perf.sum(axis=0), 0.1)})
regime_df

============================================================================

idx       params          mean_return stdev_return	10_percentile_return
0	lbk=40/10_band=2.5/1.5	0.163333	   0.0	         0.163333
```

#### Mapping the parameter siwtching strategy returns on both high and low volatility regimes 
To check if the strategy performs well across both the regimes, the code block below extracts the performance across high and low volatility
```
strat_perf_df = strat_perf.to_frame()
strat_perf_df.columns = ["strat_returns"]
strat_perf_vix_df = act_os_vix_df.join(strat_perf_df, how="left").fillna(0)
strat_perf_vix_df["VIX Close Lag"] = strat_perf_vix_df["VIX Close"].shift(1).fillna(0)
strat_perf_reg_1 = strat_perf_vix_df[strat_perf_vix_df["VIX Close Lag"]<=25]
strat_perf_reg_2 = strat_perf_vix_df[strat_perf_vix_df["VIX Close Lag"]>25]
print(f"strategy return in low volatility regime is: {strat_perf_reg_1.strat_returns.sum()} and in high volatility regime is: {strat_perf_reg_2.strat_returns.sum()}")

=============================================================================
"strategy return in low volatility regime is: 0.09899699273763563 and 
in high volatility regime is: 0.0643364071642365"
```

### Conclusion and Future Work


## References ##
    1. Stochastic Volatility Model, https://docs.pymc.io/notebooks/stochastic_volatility.html
    2. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo,
       https://arxiv.org/pdf/1111.4246.pdf
    3. Probabilistic Programmming in Python using PYMC3, https://arxiv.org/pdf/1507.08050.pdf

