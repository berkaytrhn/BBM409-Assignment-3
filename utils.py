import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def bag_of_words_ngram(data, n_number=1):
    # data parameter -> array of text samples: [text1, text2, ...]
    vct = CountVectorizer(ngram_range=(n_number, n_number))
    ngram = vct.fit_transform(data).toarray()
    return ngram, vct.get_feature_names_out()



def preprocess_text(data):
    iterable = map(lambda x:" ".join(x.split(":")[1:]) , data)
    result = np.array(pd.DataFrame(np.array([np.array(text) for text in iterable])).iloc[:,0].str.replace("[^\w\s]",""))
    return result
