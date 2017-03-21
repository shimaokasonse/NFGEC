Neural Architectures for Fine Grained Entity Type Classification
================================================================

This repository contains the source code for the experiments presented in the following research publication ([PDF](https://arxiv.org/pdf/1606.01341v1.pdf)):

    Sonse Shimaoka, Pontus Stenetorp, Kentaro Inui, Sebastian Riedel.
    "Neural Architectures for Fine Grained Entity Type Classification",
    in Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017).

## Requirements

* TensorFlow 0.11
* scikit-learn

## Preprocessing

To download and preprocess the corpora, run the following command:

	$ ./preprocess.sh

## Replicating the experiments

To run the experiments in the EACL 2017 paper, proceed as follows:

    $ python train.py figer attentive True True

You can change the  options to try different models:

	$ python train.py -h
	  	usage: train.py [-h] [--feature] [--no-feature] [--hier] [--no-hier]
	                {figer,gillick} {averaging,lstm,attentive}

		positional arguments:		    
		{figer,gillick}       dataset to train model
		{averaging,lstm,attentive}    context encoder to use in model

		optional arguments:
		-h, --help            show this help message and exit
		--feature
		--no-feature
		--hier
		--no-hier