# Models Training

The implementation is based on the [paper](https://arxiv.org/pdf/1612.01474v1.pdf) **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** It is about the uncertainty estimates obtained by using the ensemble approach proposed in the paper. 

## Dependencies
1. tensorflow
2. pandas
3. scikit-learn (+numpy)

## Description

The Models Training mainly incudes two python files:
      
1) `model.py` includes the code about the model architecture of the deep ensembles.
      
2) `train.py` includes the code to train the model and test the model.

The repository `datasets` includes an example datasets .csv used for the model training, each row in the .csv represents a training data instance.
The last clumn in the .csv is the actual cardinality of a query (in log scale).

The repository `results` includes an example of the testing results, the error is calculated by `max(est(q)/act(q), act(q)/est(q))`.


### Basic Usage

#### Example
To run the code, execute the following command from the project home directory:<br/>
	``python train.py --dataset datasets/Joins_5_12_filters_error_log2 --output results/graph_node.emd``

#### Options
You can check out the other options available to use by:<br/>
	``python train.py --help``

#### Input dataset
The supported input format is is below:

	feature_1, feature_2, ..., feature_n, <actual cardinality>
		
The features are achieved by the queries featurization. The last number in each row is the actual cardinality of a query.

#### Output
The number in each line represents the calculated error given the testing queries.
