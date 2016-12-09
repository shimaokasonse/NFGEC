# -*- coding: utf-8 -*-

import numpy as np

def create_prior(label2id_file):
    nodes = []
    num_label = 0
    with open(label2id_file) as f:
        for line in f:
            num_label += 1
            (id,label,freq) = line.strip().split()
            nodes += [label]

    prior = np.zeros((num_label,len(nodes)))
    with open(label2id_file) as f:
        for line in f:
            (id,label,freq) = line.strip().split()
            temp_ =  label.split("/")[1:]
            temp = ["/"+"/".join(temp_[:q+1]) for q in range(len(temp_))]
            code = []
            for i,node in enumerate(nodes):
                if node in temp:
                    code.append(1)
                else:
                    code.append(0)
            prior[int(id),:] = np.array(code)
    return prior


