import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils import *
from collections import defaultdict
from math import log
from tqdm import tqdm

class NaiveBayesClassifier:
    bow_spam=None
    bow_ham=None
    features=None
    n_number=None
    
    def __init__(self):
        pass

    def fit(self, X, y, n_number=1):
        self.n_number=n_number
        bow_applied, feature_names = bag_of_words_ngram(X, n_number=n_number)        
        
        print(len(feature_names))
        print(len(np.unique(feature_names)))
        print(feature_names)

        self.features = feature_names
        self.bow_spam = defaultdict(lambda : False)
        self.bow_ham = defaultdict(lambda : False)

        spams = bow_applied[y==1]
        hams = bow_applied[y==0]

        freq_spam = np.sum(spams)
        freq_ham = np.sum(hams)

        number_of_features = bow_applied.shape[1]

        freq_spam_smooth = freq_spam+number_of_features
        freq_ham_smooth = freq_ham+number_of_features

        for index in tqdm(range(len(feature_names))):
            feature = feature_names[index]
            


            freq_in_spam = np.sum(spams[:,index])
            freq_in_ham = np.sum(hams[:,index])
            
            spam_prob = None
            ham_prob = None
            
            if freq_in_spam == 0:
                freq_in_spam = 1
                spam_prob = freq_in_spam/freq_spam_smooth
            else:
                spam_prob = freq_in_spam/freq_spam

            if freq_in_ham == 0:
                freq_in_ham = 1
                ham_prob = freq_in_ham/freq_ham_smooth
            else:
                ham_prob = freq_in_ham/freq_ham
            
            if not spam_prob:
                print(spam_prob)
            if not ham_prob:
                print(ham_prob)
            self.bow_spam[feature] = spam_prob
            self.bow_ham[feature] = ham_prob
        
        # print(self.bow_spam)
        # print("*********")
        # print(self.bow_ham)

    def predict(self, X):
        results = []
        stride = self.n_number
        for sample in tqdm(X):
            sum_of_log_spam=0
            sum_of_log_ham=0
            tokens = sample.split()
            for index in range(len(tokens)-(stride-1)):
                word = " ".join(tokens[index:index+stride])

                _spam_prob = None
                _ham_prob = None
                if not self.bow_spam[word]:
                    _spam_prob = 1/len(self.features)
                else:
                    _spam_prob = self.bow_spam[word]
                if not self.bow_ham[word]:
                    _ham_prob = 1/len(self.features)
                else:
                    _ham_prob = self.bow_ham[word]
                
                
                sum_of_log_spam += np.log(_spam_prob)
                sum_of_log_ham += np.log(_ham_prob)

            #print(f"spam: {sum_of_log_spam}, ham: {sum_of_log_ham}")
            prediction = 1 if (sum_of_log_spam > sum_of_log_ham) else 0
            results.append(prediction)
        return np.array(results, dtype=np.int32)
