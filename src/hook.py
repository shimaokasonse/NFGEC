# -*- coding: utf-8 -*-
from evaluate import strict, loose_macro, loose_micro

def acc_hook(scores, y_data):
    true_and_prediction = []
    for score,true_label in zip(scores,y_data):
        predicted_tag = []
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        predicted_tag.append(lid)
        for label_id,label_score in enumerate(list(score)):
            if label_score > 0.5:
                if label_id != lid:
                    predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))
                    
    print("     strict (p,r,f1):",strict(true_and_prediction))
    print("loose macro (p,r,f1):",loose_macro(true_and_prediction))
    print("loose micro (p,r,f1):",loose_micro(true_and_prediction))
