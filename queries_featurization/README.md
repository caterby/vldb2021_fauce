Queries Featurization
======================================


## To installation each componnent, check the three repositories for details. 

Check the `requirement.txt` file to see the dependent python packes

## Description

The Queries Featurization mainly incudes three repositories:
      
1) `graph_embedding` includes the code to get the embedding results for the Tables and Columns. For the embeddings for the Tables, the input should        be an undirected join schema; for the embeddings for the Columns, the input should be either weighted or unweighted graphs for the columns                dependency.
      
2) `Joins2Vec` includes the code to get the embedding results for the Joins of a query. The input represent the information of the Join Schema.

RDC is a measure of nonlinear dependence between two (possibly
multidimensional) variables. A full description of the algorithm is given in
the [2013 paper](https://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf)
by David Lopez-Paz, Philipp Hennig, and Bernhard Schoelkopf.

## Usage

Given two NumPy arrays, `x` and `y`, the measure can be invoked as follows:

    >>> from rdc import rdc
    >>> print rdc(x, y)

If `x` and `y` are univariate, then they should be 1-D NumPy arrays; otherwise,
then should be `n`-by-`k` arrays, where `k` is the number of dimensions and `n`
is the number of examples. The two variables must have the same number of
examples, but can have different numbers of features.

There are additional keyword parameters for `rdc` that correspond to parameters
described in the paper. One new parameter is `n`, which is the number of times
the RDC is computed with different random seeds to reduce variance in the
estimation of the statistic. The median value across these `n` runs is returned.
