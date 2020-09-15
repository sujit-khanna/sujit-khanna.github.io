---
layout: post
title: Synthetic Prices for strategy backtesting and tactical trading
---

This article details, how one can generate and use synthetic prices to avoid false positive strategies, which is very commong when testing on actual historica prices. The synthetic prices are generally extracted via a Data Generating Process or DGP. There are several methods to create/train a DGP, but the most commone ones are Generative Advesarial Networks, Autoenconders (Varational and Regular), and Monte Carlo Methods. In this blog we create a DGP via Markov Chain Monte Carlo method using NUTS (or No U-Turn Sampler), and see how synthetic prices can improve out-of-sample strategy performance.

![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.
