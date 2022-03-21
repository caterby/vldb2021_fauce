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

## Usage

To see each repository for details.

