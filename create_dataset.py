# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import pickle
import numpy as np
import sys


def create_dataset(corpus_path,label2id,word2id,feature2id):
    num_of_labels = len(label2id.values())
    num_of_samples = sum(1 for line in open(corpus_path))
    storage = []
    data = np.zeros((num_of_samples,4+70+num_of_labels),"int32")
    s_start_pointer = 0
    num = 0
            
    with open(corpus_path) as f:
        for line in f:
            if len(line.split("\t")) != 5:
                continue
            (start,end,words,labels,features) = line.strip().split("\t") 
            labels, words, features = labels.split(), words.split(), features.split()
            length = len(words)
            start, end = int(start), int(end)
            labels_code = [0 for i in range(num_of_labels)]
            for label in labels:
                if label in label2id:
                    labels_code[label2id[label]] = 1
            words_code = [word2id[word] if word in word2id else word2id["unk"] for word in words]
            features_code = [feature2id[feature] for feature in features]
            storage += words_code
            data[num,0] = s_start_pointer # s_start
            data[num,1] = s_start_pointer + length # s_end
            data[num,2] = s_start_pointer + start # e_start
            data[num,3] = s_start_pointer + end  # e_end
            data[num,4:4+len(features_code)] = np.array(features_code)
            data[num,74:] = labels_code
            s_start_pointer += length
            num += 1
            if num % 100000 == 0:
                print num
    return np.array(storage,"int32"), data


def main():
    dicts = joblib.load(sys.argv[1])
    label2id = dicts["label2id"]
    word2id = dicts["word2id"]
    feature2id = dicts["feature2id"]
    storage,data = create_dataset(sys.argv[2],label2id,word2id,feature2id)
    dataset = {"storage":storage,"data":data}
    joblib.dump(dataset,sys.argv[3])

        
if(__name__=='__main__'):
    main()
