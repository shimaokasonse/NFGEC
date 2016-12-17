Neural Architectures for Fine Grained Entity Type Classification
================================================================

This repository contains the source code for the experiments presented in the following research publication ([PDF](https://arxiv.org/pdf/1606.01341v1.pdf)):

    Sonse Shimaoka, Pontus Stenetorp, Sebastian Riedel, and Kentaro Inui.
    "Neural Architectures for Fine Grained Entity Type Classification",
    in Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017), to appear.

## Requirements

* TensorFlow 0.11
* scikit-learn

## Preprocessing

To download and preprocess the corpora, run the following command:

	$ ./preprocess.sh

## Replicating the experiments

To run the experiments in the EACL 2017 paper, proceed as follows:

    $ python train.py figer attentive True True

You can change the positional options to try different models:

	$ python train.py -h
		positional arguments:
  		{figer,gillick}              dataset to train model
  		{averaging,lstm,attentive}   context encoder to use in model
  		{True,False}                 whether or not to use handcrafted features
  		{True,False}                 whether or not to use hierarchical label encoding
