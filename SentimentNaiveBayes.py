from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line=="":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory+"Positive.txt"
    (positive_list_of_lines, positive_list_of_labels)=file_reader(file_path=positive_file, label=1)

    negative_file = source_directory+"Negative.txt"
    (negative_list_of_lines, negative_list_of_labels)=file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory+"Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels)=file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set,test_labels,trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset,label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif  probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction=-1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")
    
    print("Total Sentences correctly: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions/len(test_labels)*100,5)))

    return predictions_list, round(correct_predictions/len(test_labels)*100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = []
        self.conditional = []
        self.V = []
        self.n = n_gram
        self.BOW=[]
        self.classCounts=[]
        self.D=0
        self.N=0
        self.labelmap={}

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary
    
    def create_BOW(self, training_set, training_labels, N_sentences): 
        labels = [-1, 0, 1]
        BOWS = [np.zeros((N_sentences, len(self.V)), dtype=int) for _ in range(3)]

        for i, s in enumerate(training_set):
            label = training_labels[i]
            for token in s:
                BOWS[labels.index(label)][i, self.V[token]] = 1 

        for i in range(len(BOWS)):
            BOW = BOWS[i] 
            BOWS[i] = BOW[~(BOW == 0).all(axis=1)]
        
        return BOWS  
    
    def get_class_probs(self, training_labels, N_sentences):
        counts = [0] * 3
        labelMap = {}
        for i, label in enumerate(training_labels):
            labelMap[i] = label

            if label == -1:
                counts[0] += 1
            elif label == 0:
                counts[1] += 1
            else: 
                counts[2] += 1
        
        self.classCounts = counts
        self.labelMap = labelMap

        for count in counts:
            self.prior.append(count / N_sentences)

    def train(self, training_sentences, training_labels):
        
        N_sentences = len(training_sentences)

        training_set = self.word_tokenization_dataset(training_sentences)

        self.V = self.compute_vocabulary(training_set)

        labels = [-1, 0, 1]
        lab_counts = [0, 0, 0]

        ## make 3 BOWS initialized to length Self.V * N_sentences (one per class)
        BOWS = [np.zeros((N_sentences, len(self.V)), dtype=int) for _ in range(3)]

        for i, s in enumerate(training_set):      

            # count # of labels for each class     
            label = training_labels[i]
            lab_idx = labels.index(label)
            lab_counts[lab_idx] += 1

            # set BOW index to 1 for every token in the sentence 
            for token in s:
                BOWS[lab_idx][i, self.V[token]] = 1 

        # remove all rows with only zeros 
        # makes each BOW contain only sentences that 
        for i in range(len(BOWS)):
            BOW = BOWS[i] 
            BOWS[i] = BOW[~(BOW == 0).all(axis=1)]
        
        # calculate probabilities for each class 
        for count in lab_counts:
            self.prior.append(count / N_sentences)

        self.BOW = BOWS

        for lab_inx, BOW in enumerate(self.BOW):
            # marginal probability calculation
            # diagnoal of X * X.T divided by number of entries in that class 
            divisor = lab_counts[lab_idx]
            matrix = BOW.T @ BOW
            diag = np.diag(matrix)
            probs = (list(diag / divisor))
            self.conditional.append(probs)
                                                 

    def predict(self, test_sentence):
        label_probability = {
            0: 0,
            1: 0,
            -1:0,
        }

        test_sentence = self.word_tokenization_sentence(test_sentence)

        # epislon for division by 0
        epsilon = 1e-10
        
        # iterate throguh each class 
        for lab_idx, label in enumerate([-1, 0, 1]):
            conditional = self.conditional[lab_idx]

            # start sum with the log prob of that class 
            prob = math.log(self.prior[lab_idx], 10)

            # go through each token in the sentence 
            for token, tok_idx in self.V.items():
                
                # get the marginal probability, add/subtract epilon if needed 
                prob_of_true = max(conditional[tok_idx], epsilon)
                if (prob_of_true == 1.0):
                    prob_of_true -= epsilon

                # if the token is in the sentence
                if token in test_sentence: 
                    prob += math.log(prob_of_true, 10)
                # if the token is not in the sentence, take 1 - probability 
                else: 
                    prob += math.log(1 - prob_of_true, 10)

            label_probability[label] = prob

        return label_probability

TASK = 'test' #'test'  #'train'  'test'


if TASK=='train':
    train_folder = "data-sentiment/train/"       
    training_sentences, training_labels = data_reader(train_folder)
        
    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences,training_labels)
    
    f = open('classifier.pkl', 'wb')
    pickle.dump(NBclassifier, f)
    f.close()
if TASK == 'test':
    test_folder = "data-sentiment/test/"
    test_sentences, test_labels = data_reader(test_folder)
    f = open('classifier.pkl', 'rb')
    NBclassifier = pickle.load(f)
    f.close()    
    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)

