# Graph Embedding

The *embedding* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph.

### Before running
(1) Get the Join Schema of a database, it is an undirected graph. Each node representd a Table in the database, the edges represent the PK and FK replationships among different tables. After we get the Join Schema, use it as the input for the python script to get the embedding results for each table. In Fauce, we use the IMDB dataset, the Join Schema can mannualy received.

(2) Get the global columns-dependency graph of all the columns in the database, we calculate RDC for each pair to columns in the Tables to get this graph.
The graph can be either unweighted or weighted. After we get this graph, use it as the input for the python script to get the embedding results for each columns in the database.

### Basic Usage

#### Example
To run the code, execute the following command from the project home directory:<br/>
	``python main.py --input graph/graph.edgelist --output emb/graph_node.emd``

#### Options
You can check out the other options available to use by:<br/>
	``python main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags. This impplemeation is from the Grover's KDD paper

#### Output
The first line has the following format:

	num_of_nodes dim_of_representation

The next lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by the graoh embedding method.
