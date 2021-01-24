---
layout: post
title: Regime identification using time-series clustering and its application to strategy/security allocation
---
In this post, we discuss an approach to identify and analyze different market/correlation regimes extracted by clustering time-series data of the broad US ETF market. We present interesting visuals that analyze how markets in-general behaves in each cluster/regime, and use the insights obtained from the analysis to present a strategy selection/security selection model. The following sections detail the ***clustering methodology,  results of time series clustering, (which includes broad market performance, analysis of correlation structure, network analysis for each cluster), and practical applications***. The methodology used in this post is inspired by [1] to some extent.
## Clustering Methodology

* The initial dataset consists of the top 50 most traded ETFs from which inverse ETFs are excluded to give us a total of 47 ETFs, over the period 01-02-2013 to 01-21-2021. The 47 ETFs encompass all major asset classes like Equities, Fixed Income, Credit, Commodities and FX. However equities dominate the ETF universe, the list of ETFs can be found [here](https://github.com/sujit-khanna/time-series-clustering/blob/main/data/Top50_wo_inverse.csv)

* We then split the data into monthly blocks that gives us a total of 92 non-overlapping subsets (blocks), corresponding to 92 months. Where each block contains monthly price series for each of the 47 ETFs under consideration

* We then convert the 92 non-overlapping blocks into denoised hierarchical correlation based distances. The process of creating such a correlation matrix can be briefly summaried below, and the distance metric applied on this matrix is the ***euclidean distance***

  * The first step involves calculating the price returns and creating the denoised correlation matrix using the Constant Residual Eigenvalue Method, which fits the Marchenko-Pastur (M-P) distribution to calculate the maximum theoretical eigenvalue threshold of the original correlation matrix. This threshold is used to filter out noisy eigenvalues, and the denoised correlation matrix is generated from the filtered eigenvalues and eigenvectors.

    * The next step involves adding a hierarchical structure to the correlation matrix, this approach is inspired by "Hierarchical PCA and Applications to Portfolio Management: Marco Avellaneda"[2]. Where the correlation matrix is regenerated using the natural hierarchy extracted by agglomerative hierarchical ward clustering.

    * The figure below compares three different methods of generating correlation matrices, the first sub-plot is the regular Pearson's correlation matrix, the second sub-plot represents the theory-implied correlation matrix from [3], and the last sub-plot (right) represents the hierarchical correlation matrix used in this post. As shown in the plot below the hierarchical correlation matrix has a more pronounced structure than the other two correlation matrices

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105565663-ed206c00-5cf5-11eb-8b90-1ade815044aa.png"></p>

* A Hierarchical agglomerative ward clustering technique is applied to each correlation block, to create 92 different dendrograms. Each dendrogram belongs to a particular month, which can be treated as time-series data, which can be clustered agai. 

* We can now compare hierarchical clustering belonging to different time-stamps (months), based on Dendogrammatic similarity. We define dendrogram dis/similarity using cophenetic correlation distance. This can be computed by calculating the cophenetic distance vector for each dendrogram, and then computing a correlation matrix from these cophenetic distance vectors, called cophenetic correlation matrix. The figure below depicts the cophenetic correlation matrix computed from monthly dendrograms.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105567897-79855b80-5d03-11eb-9725-01ed656116c9.png"></p>

* We cluster the cophenetic correlation matrix again and cut off the dendrogram to extract 5 flat clusters where each cluster represents a different correlation regime. (5 clusters are chosen instead of the standard choice of 3 to increase the granularity of the results). After clustering different monthly blocks, each cluster is analyzed in detail by mapping them to market behavior and network analysis.
 
 ## Results
The results analyzed in this section describe how the markets behave in each correlation regime, how correlation structure behaves in different clusters, and perform network analysis for each cluster

### Cluster Regimes vs VIX Index
One of the most rudimentary indicator of the market regime is the ***VIX Index***. Generally, during stressful market periods, the VIX index tends to rally and is subdued when there's a low risk of an adverse market event. To analyze the interaction between different clusters (regimes) and the VIX index, we plot the cluster labels for every month (corresponding to monthly dendrograms) against the VIX index, as shown below.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105568785-15b26100-5d0a-11eb-8c5c-0a81071a74cf.png", height=300></p>

Based on quick visual analysis we can see that 

* Cluster 1 generally indicates period elevated VIX levels, whereas clusters 3, and 5 indicated subdued VIX levels. Cluster 2 also indicates a period of high VIX index to some extent.

* The clusters are generally sticky/persistent, i.e. cluster labels generally do not change every month, and periods of high/low volatility regimes generally have the same cluster labels that span a few months. For example, the volatility spike (cluster label 1) at the beginning of the covid crisis spanned from Jan 2020 to May 2020, and low volatility observed (cluster label 5) just before the volmageddon event of Feb 2018 spanned from Sep 2017 to Dec 2017.


### Broad Market Performance Under Different Cluster Regimes
In this sub-section, we will map the performance of the broad market to different clusters. The broad market in our case will be the ETF dataset used in this post, the metrics used for analyzing the performance are ***mean daily return (scaled annually)*** i.e. multiply by 252, ***daily realized volatility (annualized)***, ***reward/risk or mean daily return/daily realized volatility***, ***mean correlation*** computed using fisher and inverse fisher transforms, and ***average VIX index***. These metrics are computed for each cluster by aggregating the ETF and VIX data on cluster labels. For example, the first three metrics are computed by calculating the daily returns for an equi-weighted ETF index (i.e. averaging daily returns for all ETFs daily) and aggregating the returns for each cluster label to compute the mean daily return and daily realized volatility. The same approach is used for mean correlations and average VIX index. The table below shows these metrics for each cluster sorted by ***mean daily returns***


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105569535-52815680-5d10-11eb-836d-93ec8dcd1b86.png"></p>

From the table above we can see a clear distinction between different clusters based on broad market performance.

* Cluster 1 exhibits the lowest returns and the highest daily realized volatility and VIX index, which indicates that cluster 1 corresponds to the regime when the markets are under severe stress, i.e. low return/high volatility regime

* Conversely, cluster 5 offers the highest returns with the lowest daily realized volatility and VIX index, during this regime the market rallies, and volatility is subdued.

* Clusters 2, 3, 4 falls in-between the extreme regimes, where cluster 2 is associated more to lower return and higher volatility regime, and cluster 3 and 4 offer higher returns with lower realized and implied volatility.

* Another interesting observation is that the mean correlations generally decrease as mean daily returns increase, this is synonymous with the fact that during periods of high stress all securities and asset classes exhibit higher correlation w.r.t. each other, whereas during bullish regimes the correlations between different asset classes/securities drop. This phenomenon can also be observed in the implied correlation index published daily by CBOE.

### Correlation Structure Under Different Cluster Regimes
In this subsection we analyze the correlation structure under different regimes, by computing the hierarchical correlation matrices (described above), on ETF returns grouped by different cluster labels.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105569746-5c0bbe00-5d12-11eb-8993-41f722d02c26.png"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105569759-76de3280-5d12-11eb-8824-956e4ae3faf1.png", height=180></p>

Based on the plots above we can see a clear distinction between the different clusters, which is also supported by the broad market performances observed in the previous subsection. More specifically,

* The correlation structure for cluster 1, primarily exhibits two major high correlation regions, this is supported by the fact that during high volatility periods, all asset classes and securities become highly correlated with each other. The upper region that is smaller in size consists of commodities and fixed income ETFs, whereas the lower region that dominates the matrix consists majorly of equity ETFs. However, all equity ETFs seem to behave similarly and there is no distinction between ETFs that represent different sectors/sub-sectors..

* The correlation structure for cluster 5, consists of the largest distinct correlation sub-blocks of all the clusters. Since this cluster is associated with the bull market regime, the overall correlation between different asset classes and securities is low, with correlation stronger within the smaller block of ETFs. For example, ETFs based on broad indices like S&P and Nasdaq are a part of the first (and largest) correlation block, the commodities and fixed income ETFs seems to exhibit lower correlation to all the securities. We can even observe the ETFs based on energy exhibiting the highest correlation among one another (third correlation block). Whereas the final correlation block consists of ETFs that represent emerging markets.

* As observed in the previous sub-section, the correlation structure for cluster 2 exhibits similar properties to that of cluster 1, where higher risk is associated with higher correlation between ETFs, and clusters 3 and 4 having a similar structure to that of cluster 5, where lower risk and higher returns is associated with lower correlation between ETFs.

To check how closely associated each cluster is to the other, we calculated the cophenetic correlation matrix for each cluster/regime, by computing the cophenetic distance vectors from the dendrogram of hierarchical correlation matrices for each cluster. The heatmap of the cophenetic correlation matrix is shown below.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105570269-397ba400-5d16-11eb-856f-80d23f03e17e.png"></p>

The heatmap indicates the clusters 1 and 2 are highly correlated to each other (as both represent high volatility/ low return regime), similarly clusters 3 and 5 are highly correlated with each other as they exhibit high return/ low volatility behavior

### Network Analysis of Different Cluster Regimes
Since financial markets exhibit hierarchical structure it was natural that we analyzed each cluster as a financial network. These networks will be analyzed visually as well as by their topological features. A wide variety of networks can be used to model financial markets like threshold networks, causal networks, knowledge graphs, etc. However, in this sub-section, we will create a minimum spanning tree (MST) from the hierarchical correlation matrices for each cluster.
The plot below, shows the MST constructed for each cluster/regime

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105570835-89a83580-5d19-11eb-852d-e3c8f1e4f519.png"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105570848-a80e3100-5d19-11eb-9e02-278a4b5b913d.png", height=180></p>


From the plot above we can observe that 

* MST for clusters 1 and 2 shrinks, whereas for clusters 3,4, and 5 MST expands. Also note that for cluster 5, we can observe three distinct regions in the MST. This is in-line with existing research that states that MST strongly shrinks during a market crisis, and expands during periods of high returns and low volatility.

* For clusters 1, SHY ETF i.e 1-3 Year Treasury Bond ETF, has the most number of edges indicating that this ETF dominates during the period of market crisis (which follows economic rationale), similarly for cluster 2 which is also a low return/high volatility clusters has the most number of connections to TLT ETF, i.e. iShares 20+ Year Treasury Bond ETF, which is a longer maturity ETFF

* Conversely, for bullish regime clusters, we observe ETFs in either Credit/Equity/Forex/Comdty space with the most number of edge connections, i.e. for cluster 4 most number of connections come from the LQD ETF which is iBoxx USD Investment Grade Corporate Bond ETF. In cluster 5 highest number of edges are directed to/from XLP consumer staples S&P500 ETF and GLD Gold Spot ETF.

We will now analyze topological features of the MST for each cluster, the topological features used are,

* ***Average Shortest Path Length (ASPL)***:As the name suggests it averages the shortest path length between two nodes for the entire graph (MST)A

* ***Diameter***: This feature represents the diameter of the MST for each cluster

* ***Number of Communities (NC)***: For each cluster this represents the community of stocks that are closely related to each other

* ***Betweenness Centrality (BC)*** : Betweenness centrality of a node 𝑣 is the sum of the fraction of all-pairs shortest paths that pass through 𝑣. For the graph level analysis we will average this value for all the nodes

* ***Closeness Centrality (CC)***: It is reciprocal of the average shortest path distance to u over all n-1 reachable nodes, higher values of closeness indicate higher centrality. As before we will average this value for all the nodes in the graph


* ***Eigenvector Centrality (EC)***: The eigenvector centrality computes the centrality for a node based on the centrality of its neighbors, which will be average for all the nodes for graph level analysis. EC can alos be a measure of influence of a particular node in a network

The table below contains the topological feature values for all 5 clusters

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105614319-a3d42900-5d96-11eb-9be5-89504fa63eeb.png"></p>

From the table, it is evident that MSTs of clusters 1 and 2 are highly condensed exhibited by low *ASPL*, *BC*, and high *CC* feature values. Clusters 3 and 5 seem to exhibit the most stretched out MSTs based on these topological features. One interesting observation emanating from the table above is that *** cluster 3 exhibits the lowest *EC* score, which might indicate that this cluster/regime is the most diversified, that is not dominated by few ETFs ***.

The last table in this section looks at nodes that dominate the MST with the most influencing feature values, in this table we add another feature called ***Degree*** that indicates the ETF that has the highest number of edge connections. In the table below the ETFs/nodes are chosen based on the highest feature values.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105614650-9ddf4780-5d98-11eb-8029-dfae79aef763.png"></p>

This table confirms the results from the visual analysis of the MST for all clusters,

* In the high volatility/low return cluster fixed income products are the most influential ETFs which follows the economic rationale

* For regime 4, Credit ETF LQD dominates the MST, whereas for bull market regimes 3 and 5 Forex ETF (FXE), and Gold & Equity ETFs (i.e. GLD and VTI) seem to be the most influential ETFs

## Potential Applications
Since all the analysis done in this post was not predictive, the results cannot be directly applied to create a new trading strategy or improve an existing one. One straightforward application of such a time-series clustering approach is to use it for Asset, Security, or Strategy allocation/selection. 

There can be a couple of ways to approach this problem, the first approach is to predict the cluster label based on features of the hierarchical correlation matrix or MST like cophenetic distance, MST centrality measures, spectral features, etc. using a classifier like Random Forest or SVM. The second approach can take advantage of the fact that cluster labels tend to be sticky/persistent so one can assume that the markets will belong to the same cluster regime for the next time period. Then one can compute the performance of different ETFs/Strategies for the most recent cluster label in the data, and filter out the ETFs/Strategies that performed poorly in this most recent cluster label. One can even create a Market Neutral portfolio based on cross-sectional factors over this cluster regime or use a rank-weighted allocation scheme (with ETFs/Strategies ranked over this cluster label).

```Another post will be added shortly to the blog to showcase this application using the second approach```

## Conclusion
To conclude this post, we defined a framework to cluster the broad market time-series data using cophenetic correlation distance extracted from the dendrograms non-overlapping monthly returns. To assess if different cluster labels are indicative of different market/correlation regimes we first mapped the broad market performance in each cluster label and found that clusters 1 and 2 indicate low returns/high volatility regime, whereas clusters 3 and 5 indicate periods of high returns/low volatility. We further analyzed the correlation structure for each cluster label, which corroborated our initial results, where low returns/high volatility cluster labels exhibited a few large blocks of highly correlated ETFs, whereas cluster labels 3 and 5 had a smaller but greater number of blocks with highly correlated ETFs. Finally, we performed network analysis using MST to identify the ETFs that dominate each cluster labels. We found that in the low return/high volatility clusters fixed income ETFs were the influencing players, whereas in other clusters Gold, Credit, Forex, and Equity ETFs were more influencing.

## References ##
    1. Handling risk-on/risk-off dynamics with correlation regimes and correlation networks: Jochen Papenbrock, Peter Schwendner
    2. Hierarchical PCA and Applications to Portfolio Management: Marco Avellaneda
    3. Estimation of Theory-Implied Correlation Matrices: Marcos Lopez de Prado
