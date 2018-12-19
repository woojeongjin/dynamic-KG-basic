Code framework for KG completion.
Dataset: this dataset comes from Know-Evolve repo.
	train_500.txt, test_500.txt, stat_500.txt - small dataset (500 entities)
	train.txt, test.txt, stat.txt - original ICEWS dataset
	1st column in train.txt - subject entity
	2nd column - relation
	3rd column - object entity
	4th column - time

	1st figure in stat.txt - number of entities
	2nd figure in stat.txt - number of relations

data.py: this is for corrupting triples and other functions for data

util.py: this is collection of frequent functions

evaluation.py: evaluation codes

train.py: train codes

You can run the code with
	python train.py

But you have to implement your model first.# dynamic-KG-basic
