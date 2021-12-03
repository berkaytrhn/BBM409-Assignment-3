import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils import *


class NaiveBayesClassifier:
    bow_positive=None
    bow_negative=None
    features=None
    
    def __init__(self):
        pass

    def fit(self, X, y, n_number=1):
        bow_applied, feature_names = bag_of_words_ngram(X, n_number=n_number)        
        
        print(feature_names)

        self.features = feature_names
        self.bow_positive = dict()
        self.bow_negative = dict()

        positives = bow_applied[y==1]
        negatives = bow_applied[y==0]

        

        for index, feature in enumerate(feature_names):
            print(positives[:,index])
            print(positives[:,index] == 0)
            zero_positives = positives[positives[:,index] == 0]
            print(zero_positives)
            print("******************************")

            print(positives[:,index])
            print(positives[:,index] == 1)
            one_positives = positives[positives[:,index] == 1]
            print(one_positives)
            exit()

            # [number of positives when feature value == 0, number of positives when feature value == 1],
            #  this way it can be accessed by value as index
            self.bow_positive[feature] = []

    def predict(self, X):
        pass