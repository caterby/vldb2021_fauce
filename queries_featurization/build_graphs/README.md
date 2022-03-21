RDC: Randomized Dependence Coefficient
======================================

The RDC is a measure of nonlinear dependence between two variables. In Fauce, we use RDC to construct the local and global columns-dependency graph for each Table and the database respectively. This method is decribed in the NIPS 2013 paper by David Lopez-Paz, etc.  

## Installation Instructions

To install, run:

    $ [sudo] python setup.py install

## Usage

Given two NumPy arrays, `x` and `y`, the measure can be invoked as follows:

    >>> from rdc import rdc
    >>> print rdc(x, y)

In Fauce, `x` and `y` represent a pair of clumns with real-valued numbers in a Table.

## Example

We include two examples, `full_data_12_19_new_CHKK_cluster1.csv` and `full_data_12_19_new_zika_cluster1.csv`. The output of the columns dependency of this two files are contained in the files `matrix_12_19_CHKK_cluster1.csv` and `matrix_12_19_zika_cluster1.csv` respectively. The dependency is represented as a matrix, each element `matrix[i][j]` represents the dependency of Columns `i` and `j` in a Table. 
