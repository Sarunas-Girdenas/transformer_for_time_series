# TST - Transformer Architecture for Time Series Data

This is Transformer for time series classification. Very heavily inspired by [Peter Bloem's](http://peterbloem.nl/blog/transformers) code and explanations. Idea of adding positional encodings with 1D convolutions is from [Attend and Diagnose](https://arxiv.org/abs/1711.03905) paper.

## The Problem
Given sequence of time series, determine to which class it belongs. In the financial context this would be something like "Can we predict if the future price will go up or down given the sequence of last `n` prices?".

## Approach
Instead of using something like LSTM, RNN or TCN, we've decided to build Transformer. To start with, [Medium](https://towardsdatascience.com/attention-for-time-series-classification-and-forecasting-261723e0006d) has a great review of various methods.

We've mostly used approach from Attend & Diagnose paper; Dense Interpolation is taken from [here](https://github.com/khirotaka/SAnD/blob/master/core/modules.py). See chart below for very high level archicture overview:

![Architecture](https://miro.medium.com/max/1400/1*eZQZel7w-Ukp7oOtXuocJg.png)

_Image taken from [here](https://towardsdatascience.com/attention-for-time-series-classification-and-forecasting-261723e0006d)._

## Sample Usecase
See `transformers_time_series.ipynb` for an example. 
