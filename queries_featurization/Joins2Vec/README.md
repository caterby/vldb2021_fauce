# Joins2vec

This repository contains the "tensorflow" implementation of our paper "Joins2vec". The implementation is from the paper: subgraph2vec: Learning Distributed Representations of Rooted Sub-graphs from Large Graphs.
This code is developed in python 2.7. It is ran and tested on Ubuntu 18.04 and 20.04.
The input of the this method is the join schame of a database, which is represented as an undirected graph.
Each join relationship is represented as a subgraph in the join schema.

##### Before running
Install the python packages mentioned in the `requirement.txt` file:


#####  Obtain the rooted Join vectors using subgraph2vec:
	1. move to the folder "Joins2Vec" (command: cd Joins2Vec)
	2. make sure that the information if the join schema is available in the same rep'../example_dataset/datasets/dir_graphs/')
	3. run main.py --corpus <dataset of graph files> --class_labels_file_name <file containing class labels>:
		*Generate the weisfeiler-lehman kernel's rooted subgraphs from all the graphs 
		*Train skipgram model to learn Joins embeddings. 
	3. example: 
		*python main.py --corpus ../example_data/datasets/node_edges --class_labels_file_name ../example_data/datasets/node.Labels
	

#### Other command line args:
	optional arguments:
		-h, --help            show this help message and exit
		-c CORPUS, --corpus CORPUS
				        Path to directory containing graph files to be used
				        for graph classification or clustering
		-l CLASS_LABELS_FILE_NAME, --class_labels_file_name CLASS_LABELS_FILE_NAME
				        File name containg the name of the sample and the
				        class labels
		-o OUTPUT_DIR, --output_dir OUTPUT_DIR
				        Path to directory for storing output embeddings
		-b BATCH_SIZE, --batch_size BATCH_SIZE
				        Number of samples per training batch
		-e EPOCHS, --epochs EPOCHS
				        Number of iterations the whole dataset of graphs is
				        traversed
		-d EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
				        Intended subgraph embedding size to be learnt
		-neg NUM_NEGSAMPLE, --num_negsample NUM_NEGSAMPLE
				        Number of negative samples to be used for training
		-lr LEARNING_RATE, --learning_rate LEARNING_RATE
				        Learning rate to optimize the loss function
		--n_cpus N_CPUS       Maximum no. of cpu cores to be used for WL kernel
				        feature extraction from graphs
		--wlk_h WLK_H         Height of WL kernel (i.e., degree of rooted subgraph
				        features to be considered for representation learning)
