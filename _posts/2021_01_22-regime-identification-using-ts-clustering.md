---
layout: post
title: Regime identification using time-series clustering and its application to strategy/security allocation
---
In this post we discuss an approach to identify and analyze different market/correlation regimes extracted by clustering time-series data of the broad US ETF market. We present interesting visuals that analyze how markets in-general behaves in each regime, and use the insights obtained from the analysis to create a strategy selection/security selection model based on different correlation regimes. The following sections details the **methodology used, results from time series clustering, network analysis of each regime and its practical applications.**. The methodology used in this post is inspired from [1] to some extent.

## Clustering Methodology

* The initial dataset consists of top 50 most traded ETFs from which  inverse ETFs are excluded to give us a total of 47 ETFs, over a period of 01-02-2013 to 01-21-2021. The 47 ETFs encompass all major asset classes like Equities, Fixed Income, Commodities and FX, however equities dominate the ETF universe. The list of ETFs can be found here (insert link.)

* We then split the data into monthly blocks that gives us a total of 92 non-overlapping subsets (blocks), corresponding to 91 months. Where each block contains monthly price series for each of the 47 ETFs under consideration

* We then convert the 91 non-overlapping blocks into denoised hierarchical correlation based distances. The process of creating such a correlation matrix can be briefly summaried below, and the distance metric applied on this matrix is the ***euclidean distance***

  * The first step involves calculating the price returns and creating the denoised correlation matrix using the Constant Residual Eigenvalue Method, that fits the Marcenko-Pastur (M-P) distribution to calculate the maximum theoretical eigenvalue threshold of the original correlation matrix. This threshold is used to filter out noisy eigenvalues, and the denoised correlation matrix is generated from the filtered eigenvalues and eigenvectors.

    * The next step involves adding a hierarchical structure to the correlation matrix, this approach is inspired by "Hierarchical PCA and Applications to Portfolio Management: Marco Avellaneda"[2]. Where the correlation matrix is regenerated using the natural hierarchy extracted by agglomerative hierarchical ward clustering.

    * The figure below compares three different methods of generating correlation matrics, the first sub-plot is the regular pearson's correlation matrix, second sub-plot represents the theory-implied correlation matrix from *Marcos Lopez de Prado*, and the last sub-plot (right) represents the hierarchical correlation matrix  used in this post. As shown in the plot below the hierarchical correlation matrix has more pronounced structure than other two correlation matrices.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105565663-ed206c00-5cf5-11eb-8b90-1ade815044aa.png"></p>

* A Hierarchical agglomerative ward clustering technique is applied to each correlation blocks, to create 91 different dendograms. Each dendogram belongs to a particular month, which can be treated as time-series data, which can be clustered again. 

* We can now compare hierarchical clustering belonging to different time-stamps (months), based on dendogrammatic similarity. We define dendogram dis/similarity using cophenetic correlation distance. This can be computed by calculating cophenetic distance vector for each dendogram, and then computing a correlation matrix from the cophenetic distance vectors, called cophenetic correlation matrix. The figure below depicts the cophenetic correlation matrix computed from monthly dendograms.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105567897-79855b80-5d03-11eb-9725-01ed656116c9.png"></p>

* We cluster the cophenetic correlation matrix again, and cut off the dendogram to extract 5 flat clusters where each cluster represents a different correlation regime. (5 clusters are choosen instead of standard choice of 3 to increase the granularity of the results). After clustering different monthly blocks, each cluster is analyzed in detail by mapping them to market behavior and network analysis.
 
 ## Results
The results analyzed in this section  describe how the markets behave in each correlation regime. 

### Cluster Regimes vs VIX Index
One of the most basic indicator of market regime is the ***VIX Index***. Generally during stressful market periods VIX index tends to rally, and is subdued when there's low risk of an adverse market event. To analyze the interaction between different clusters (regimes) and the VIX index, we plot the cluster labels for every month (corresponding to monthly dendograms) against the VIX index, as shown below.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105568785-15b26100-5d0a-11eb-8c5c-0a81071a74cf.png"></p>

Based on quick visual analysis we can see that 

* cluster 1 generally indicates period elevated VIX levels, where as clusters 3, 4, and 5 indicated subdued VIX levels. Cluster 2 also indicates period of high VIX index to some extent.

* The clusters are generally sticky/persistant, i.e. cluster lables generally do not change every month, and periods of high/low volatility regimes generally have same cluster lables that span a few months. For example the volatility spike (cluster label 1) at the beginning of covid crisis spanned from Jan, 2020 to May 2020, and low volatility observed (cluster label 5) just before the volmageddon event of Feb 2018 spanned from Sep, 2017 to Dec, 2017.


### Broad Market Performance Under Different Cluster Regimes
In this sub-section we will map the performance of the broad market to different clusters. The broad market in our case will be the ETF dataset used in this post, the metrics used for analyzing the performance are ***mean daily return (scaled annually)*** i.e. multiply by 252, ***daily realized volatility (annualized)***, ***reward/risk or mean daily return/daily realized volatility***, ***mean correlation*** computed using fisher and inverse fisher transforms, and ***average VIX index***. These metrics are computed for each cluster by aggregating the ETF and VIX data on cluster labels. For example the first three metrics are computed by calculating the daily returns for an equi-weighted ETF index (i.e. averaging daily returns for all ETFs daily) and aggregating the returns for each cluster label to compute the mean daily return and daily realized volatility. The same approach is used for mean correlations and average vix index. The table below shows these metrics for each cluster sorted by ***mean daily returns***


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105569535-52815680-5d10-11eb-836d-93ec8dcd1b86.png"></p>

From the table above we can see a clear distinction between different clusters based on broad market performance.

* Cluster 1 exhibits the lowest returns and the highest daily realized volatility and VIX index, which shows that cluster 1 corresponds to the regime when the markets are under sever stress, which can also be indicated as low return, high volatility regime (as indicated by the metrics)

* Conversely cluster 5 offers highest returns with the lowest daily realized volatilty and VIX index, these are periods where the market rallies and volatility is subdued.

* Clusters 2, 3, 4 fall in-between the extreme regimes, where cluster 2 is associated more to lower return and higher volatility regime, and cluster 3 and 4 offer higher returns with lower realized and implied volatilty.

* Another interesting observation is that the mean correlations monotonically decreases as mean daily returns increase, this is synonymous to the fact that during periods of high stress all securities and asset classes exhibit higher correlation w.r.t. each other, whereas during bullish regimes the correlations between different asset classes/securities drops. This phenomenon can also be observed in the implied correlation index published daily by CBOE.

### Correlation Structure Under Different Cluster Regimes
In this subsection we analyze the correlation structure under different regimes, by computing the hierarchical correlation matrices (described above), on ETF returns grouped by different cluster labels.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105569746-5c0bbe00-5d12-11eb-8993-41f722d02c26.png"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105569759-76de3280-5d12-11eb-8824-956e4ae3faf1.png", height="180"></p>

Based on the plots above we can see a clear distinction between the different clusters, which is also supported by the broad market performances observed in the previous subsection. More specifically,

* Correlation structure for cluster 1, primarily exhibits two major high correlation regions, this is supported by the fact that during high volatility periods, all asset classes and securities become highly correlated with each other. The upper region that is smaller in size consists of commodities and fixed income ETFs, where as the lower region that dominates the matrix consist majorly of equity ETFs. However all equity ETFs seem to behave similarly and there is no distinction between  ETFs that represent different sectors/sub-sectors.

* Correlation structure for cluster 5, consists of the largest distinct correlation sub-blocks of all the clusters. Since this cluster is associated with bull market regime, the overall correlation between different asset classes and securities is low, with correlation stronger within smaller block of ETFs. For example ETFs based on broad indices like S&P and Nasdaq are a part of the first (and largest) correlation block, the commodities and fixed income ETFs seems to exhibit lower correlation to all the securities. We can even observe the ETFs based on energy exhibiting highest correlation among one another (third correlation block). Where as the final correlation block consists of ETFs that represents emerging markets.

* As observed in the previous sub-section, the correlation structure for cluster 2 exhibits similar properties to that of cluster 1, where higher risk is associated with higher correlation between ETFs, and clusters 3 and 4 having a similar structure to that of cluster 5, where lower risk and higher returns is associated with lower correlation between ETFs.

To check how closely associated each cluster is to other, we calculated the cophenetic correlation matrix for each cluster/regime, by computing the cophenetic distance vectors from the dendogram of hierarchical correlation metrix for each cluster. The heatmap of the cophenetic correlation matrix is shown below.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105570269-397ba400-5d16-11eb-856f-80d23f03e17e.png", , height="180"></p>

The heatmap indicates the clusters 1 and 2 are highly correlated to each other (as both represent high vol/ low return behavior), similarly cluster 3 and 5 exhibit sort of higher correlation as they exhibit high return/ low volatility behavior.

### Network Analysis of Different Cluster Regimes
Since financial markets exhibit hierarchical structure it was natural that analyze each cluster as a financial network. These networks will be analyzed visually as well as by its topological features. A wide variety of networks can be used to model financial markets like threshold nework, causal networks, knowledge graphs etc. However in this section we will create a minimum spanning tree (MST) from the hierarchical correlation matrices for each cluster, such that only the highest correlated connections are chosen.
The plot below, shows the MST constructed for each cluster/regime

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105570835-89a83580-5d19-11eb-852d-e3c8f1e4f519.png",height="200"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105570848-a80e3100-5d19-11eb-9e02-278a4b5b913d.png", height="180"></p>


From the plot above we can observe that 

* MST for clusters 1 and 2 shrinks, where as for clusters 3,4, and 5 it expands. Also note that for cluster 5, we can observe three seperate regions in the MST, that are grouped together. This is in-line with existing research that states that MST strongly shrinks during market crisis, and espands during periods of high returns and low volatility.

* For clusters 1, SHY ETF i.e 1-3 Year Treasury Bond ETF, has the most number of edges indicating that this ETF dominates during the period of market crisis (which follows economic rationale), similarly for cluster 2 which is also a low return/high volatility clusters has most number of connections to TLT ETF, i.e. iShares 20+ Year Treasury Bond ETF, which is a longer maturity ETF

* Conversly for bullish regime clusters, we observe ETFs in either Credit/Equity/Forex/Comdty space with the most number of edge connections, i.e. for cluster 4 most number of connections come from the LQD ETF which is iBoxx USD Investment Grade Corporate Bond ETF. In cluster 5 highest number of edges are directed to/from XLP consumer staples S&P500 ETF and GLD Gold Spot ETF.

We will now analyze topological features of the MST for each cluster, the topological features used are,

* ***Average Shortest Path Length (ASPL)***:As the name suggests it averages the shortest path length between two nodes for the entire graph (MST)

* ***Diameter***: This feature represents the diameter of the MST for each cluster

* ***Number of Communities (NC)***: For each cluster this represents the community of stocks that are closely related to each other

* ***Betweenness Centrality (BC)*** : Betweenness centrality of a node 𝑣 is the sum of the fraction of all-pairs shortest paths that pass through 𝑣. For the graph level analysis we will average this value for all the nodes

* ***Closeness Centrality (CC)***: It is reciprocal of the average shortest path distance to u over all n-1 reachable nodes, higher values of closeness indicate higher centrality. As before we will average this value for all the nodes in the graph


* ***Eigenvector Centrality (EC)***: The eigenvector centrality computes the centrality for a node based on the centrality of its neighbors, which will be average for all the nodes for graph level analysis. EC can alosbe a measure of influence of a particular node in a network

The table below contains the topological feature values for all 5 clusters

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105614319-a3d42900-5d96-11eb-9be5-89504fa63eeb.png"></p>

From the table it is evident that MSTs of clusters 1 and 2 are highly condensed exhibited by low *ASPL*, *BC*, and high *CC* feature values. Clusters 3 and 5 seem to exhibit the most stretched out MSTs based on this topological features. One interesting observation emenating from the table above is that cluster 3 exhibits the lowest *EC* score, which might indicate that this cluster/regime is the most diversified, that contains the least number of dominating factors.

The last table in this section looks at nodes that dominate the MST with the most influencing feature values, in this table we add another feature called ***Degree*** that indicates the ETF that has the highest number of edge connections. In the table below the ETFs/nodes are chosen based on the highest feature values.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/105614650-9ddf4780-5d98-11eb-8029-dfae79aef763.png"></p>

This table confirms the results from the visual analysis of the MST for all clusters,

* In high volatility/low return cluster fixed income products are the most influential ETFs which follows economic rationale

* For regime 4, Credit ETF LQD dominates the MST, whereas for bull market regimes 3 and 5 Forex ETF (FXE), and Gold & Equity ETFs (i.e. GLD and VTI) seem to be the most influentual ETFs


## Conclusion
