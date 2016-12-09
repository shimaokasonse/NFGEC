# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import pickle
import numpy as np
import sys

def load_X2id(file_path):
    X2id = {}
    id2X = {}
    with open(file_path) as f:
        for line in f:
            temp = line.strip().split()
            id,X = temp[0],temp[1]
            X2id[X] = int(id)
            id2X[int(id)] = X
    return X2id, id2X

def load_word2vec(file_path):
    word2vec = {}
    with open(file_path) as lines:
        for line in lines:
            split = line.split(" ")
            word = split[0]
            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            word2vec[word] = np.array(vector)
    return word2vec

def create_id2vec(word2id,word2vec):
    unk_vec = word2vec["unk"]
    dim_of_vector = len(unk_vec)
    num_of_tokens = len(word2id)
    id2vec = np.zeros((num_of_tokens,dim_of_vector))
    for word,t_id in word2id.items():
        id2vec[t_id,:] = word2vec[word] if word in word2vec else unk_vec
    return id2vec



def main():
    print "word2id..."
    word2id, id2word = load_X2id(sys.argv[1])
    print "feature2id..."
    feature2id, id2feature = load_X2id(sys.argv[2])
    print "label2id..."
    label2id, id2label = load_X2id(sys.argv[3])
    print "word2vec..."
    word2vec = load_word2vec(sys.argv[4])
    print "id2vec..."
    id2vec = create_id2vec(word2id,word2vec)
    print "done!"
    dicts = {"id2vec":id2vec,"word2id":word2id,"id2word":id2word,"label2id":label2id,"id2label":id2label,"feature2id":feature2id,"id2feature":id2feature}
    print "dicts save..."
    joblib.dump(dicts,sys.argv[5])

        
if(__name__=='__main__'):
    main()
