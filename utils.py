import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words_ngram(data, n_number=1):
    # data parameter -> array of text samples: [text1, text2, ...]
    vct = CountVectorizer(ngram_range=(n_number, n_number))
    ngram = vct.fit_transform(data).toarray()
    return ngram, vct.get_feature_names()



def preprocess_text(data):
    iterable = map(lambda x:" ".join(x.split(":")[1:]) , data)
    return np.array([np.array(text) for text in iterable])