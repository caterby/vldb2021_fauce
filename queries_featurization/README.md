Queries Featurization
======================================


## To installation each componnent, check the three repositories for details. 

Check the `requirement.txt` file to see the dependent python packes

## Description

The Queries Featurization mainly incudes three repositories:
      
1) `graph_embedding` includes the code to get the embedding results for the Tables and Columns. For the embeddings for the Tables, the input should        be an undirected join schema; for the embeddings for the Columns, the input should be either weighted or unweighted graphs for the columns                dependency.
      
2) `Joins2Vec` includes the code to get the embedding results for the Joins of a query. The input represent the information of the Join Schema.

3) `build_graphs` includes the code to build the columns-dependency graphs, it is based on the Randomized Dependence Coefficient (RDC) method to get the dependencies among columns. Columns are represented nodes, the weight among two columns can be represented as the calculated dependency value by the RDC method.


## Usage

To see each repository for details.
