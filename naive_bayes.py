import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils import *
from collections import defaultdict
from math import log

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
        
        print(feature_names)

        self.features = feature_names
        self.bow_spam = defaultdict(lambda x:False)
        self.bow_ham = defaultdict(lambda x:False)

        spams = bow_applied[y==1]
        hams = bow_applied[y==0]

        for index, feature in enumerate(feature_names):

            #print("********")
            #print(np.sum(spams[:,index]))
            #print(f"{feature} -> {np.sum(spams[:,index])/np.sum(spams)}")
            self.bow_spam[feature] = np.sum(spams[:,index])/np.sum(spams)
            self.bow_ham[feature] = np.sum(hams[:,index])/np.sum(hams)
        
        # print(self.bow_spam)
        # print("*********")
        # print(self.bow_ham)

    def predict(self, X):
        print(X)
        vectorized, features = bag_of_words_ngram(X, self.n_number)
        for sample in vectorized:
            print(sample)
            
            