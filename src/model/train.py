# -*- coding: utf-8 -*-

import sys
import random
random.seed(123)
sys.path.append('../../')
sys.path.append('../')
from sklearn.externals import joblib
from batcher import Batcher
from hook import acc_hook

data_name = "gillick"

print "loading dataset..."
train_dataset = joblib.load("../../data/"+data_name+"_train.pkl")
dev_dataset = joblib.load("../../data/"+data_name+"_dev.pkl")
test_dataset = joblib.load("../../data/"+data_name+"_test.pkl")

dicts = joblib.load("../../data/dicts_"+data_name+".pkl")

print "train_size:", train_dataset["data"].shape[0]
print "dev_size: ", dev_dataset["data"].shape[0]
print "test_size: ", test_dataset["data"].shape[0]

train_batcher = Batcher(train_dataset["storage"],train_dataset["data"][:],1000,10,dicts["id2vec"])
dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],dev_dataset["data"].shape[0],10,dicts["id2vec"])
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],test_dataset["data"].shape[0],10,dicts["id2vec"])

from nn_model import Model
model = Model(type = "gillick", encoder = "averaging", hier = True, feature = False)


for epoch in range(200):
    train_batcher.shuffle()
    print "epoch",epoch
    for i in range(10):
        context_data, mention_representation_data, target_data, feature_data = train_batcher.next()
        #model.train(context_data, mention_representation_data, target_data, feature_data)
        model.train(context_data, mention_representation_data, target_data)

    print "------dev--------"
    context_data, mention_representation_data, target_data, feature_data = dev_batcher.next()
    #scores = model.predict(context_data, mention_representation_data,feature_data)
    scores = model.predict(context_data, mention_representation_data)
    acc_hook(scores, target_data)
    print "-----test--------"
    context_data, mention_representation_data, target_data, feature_data = test_batcher.next()
    #scores = model.predict(context_data, mention_representation_data, feature_data)
    scores = model.predict(context_data, mention_representation_data)
    acc_hook(scores, target_data)
