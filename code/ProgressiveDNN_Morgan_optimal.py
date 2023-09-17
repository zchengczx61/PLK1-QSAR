from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
np.random.seed(123)

import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

dataset_file = 'train1a.csv'
task = ['pic50']
feature_field = 'smiles'
dataset_file1 = 'valid1.csv'
dataset_file2 = 'test1.csv'

featurizer_func = dc.feat.CircularFingerprint(size=1024)

loader = dc.data.CSVLoader(tasks=task, feature_field='smiles', featurizer=featurizer_func)

train_dataset = loader.create_dataset(dataset_file)
transformers = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
]
for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)

valid_dataset = loader.create_dataset(dataset_file1)
transformers1 = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=valid_dataset)
]
for transformer in transformers1:
    valid_dataset = transformer.transform(valid_dataset)
    
test_dataset = loader.create_dataset(dataset_file2)
transformers2 = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=test_dataset)
]
for transformer in transformers2:
    test_dataset = transformer.transform(test_dataset)

learning_rate = 0.001
model = dc.models.ProgressiveMultitaskRegressor(
    n_tasks=1,
    n_features=1024,
    dropouts=0.5,
    layer_sizes=[500],
    learning_rate=0.0001,
    batch_size=32)
model.fit(train_dataset, nb_epoch=10)

r2_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
rms_metric = dc.metrics.Metric(dc.metrics.rms_score)
mae_metric = dc.metrics.Metric(dc.metrics.mae_score)
metrics = [r2_metric, rms_metric, mae_metric]    
    
print("Training set score:", model.evaluate(train_dataset, metrics, transformers))
print("valid set score:", model.evaluate(valid_dataset, metrics, transformers1))
print("Test set score:", model.evaluate(test_dataset, metrics, transformers2))
