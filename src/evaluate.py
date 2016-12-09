import sys


def f1(p,r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )

def strict(true_and_prediction):
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    precision = recall = correct_num / num_entities 
    return precision, recall, f1( precision, recall)

def loose_macro(true_and_prediction):
    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1( precision, recall)

def loose_micro(true_and_prediction):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels))) 
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1( precision, recall)


if __name__ == "__main__":
    file = open(sys.argv[1])
    true_and_prediction = []
    for line in file:
        temp = line.split("\t")
        if len(temp) == 1:
            true_labels = temp[0].split()
            predicted_labels = []
        else:
            true_labels, predicted_labels = temp
            true_labels = true_labels.split()
            predicted_labels = predicted_labels.split()
            true_and_prediction.append((true_labels,predicted_labels))
    #for each in true_and_prediction:
        #print(each)
    print("     strict (p,r,f1):",strict(true_and_prediction))
    print("loose macro (p,r,f1):",loose_macro(true_and_prediction))
    print("loose micro (p,r,f1):",loose_micro(true_and_prediction))
    file.close()
