# -*- coding: utf-8 -*-
import argparse
from sklearn.externals import joblib
from src.model.nn_model import Model
from src.batcher import Batcher
from src.hook import acc_hook

parser = argparse.ArgumentParser()
parser.add_argument("dataset",help="dataset to train model",choices=["figer","gillick"])
parser.add_argument("encoder",help="context encoder to use in model",choices=["averaging","lstm","attentive"])
parser.add_argument("feature",help="whether or not to use handcrafted features",type=bool,choices=[True,False])
parser.add_argument("hier",help="whether or not to use hierarchical label encoding",type=bool,choices=[True,False]) 
args = parser.parse_args()

print "Creating the model"
model = Model(type=args.dataset,encoder=args.encoder,hier=args.hier,feature=args.feature)


print "Loading the dictionaries"
d = "Wiki" if args.dataset == "figer" else "OntoNotes"
dicts = joblib.load("data/"+d+"/dicts_"+args.dataset+".pkl")

print "Loading the datasets"
train_dataset = joblib.load("data/"+d+"/train_"+args.dataset+".pkl")
dev_dataset = joblib.load("data/"+d+"/dev_"+args.dataset+".pkl")
test_dataset = joblib.load("data/"+d+"/test_"+args.dataset+".pkl")

print 
print "train_size:", train_dataset["data"].shape[0]
print "dev_size: ", dev_dataset["data"].shape[0]
print "test_size: ", test_dataset["data"].shape[0]

print "Creating batchers"
# batch_size : 1000, context_length : 10
train_batcher = Batcher(train_dataset["storage"],train_dataset["data"],1000,10,dicts["id2vec"])
dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],dev_dataset["data"].shape[0],10,dicts["id2vec"])
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],test_dataset["data"].shape[0],10,dicts["id2vec"])

step_par_epoch = 2000 if args.dataset == "figer" else 150

print "start trainning"
for epoch in range(5):
    train_batcher.shuffle()
    print "epoch",epoch
    for i in range(step_par_epoch):
        context_data, mention_representation_data, target_data, feature_data = train_batcher.next()
        model.train(context_data, mention_representation_data, target_data, feature_data)
        
    print "------dev--------"
    context_data, mention_representation_data, target_data, feature_data = dev_batcher.next()
    scores = model.predict(context_data, mention_representation_data,feature_data)
    acc_hook(scores, target_data)

print "Training completed.  Below are the final test scores: "
print "-----test--------"
context_data, mention_representation_data, target_data, feature_data = test_batcher.next()
scores = model.predict(context_data, mention_representation_data, feature_data)
acc_hook(scores, target_data)

print "Cheers!"
