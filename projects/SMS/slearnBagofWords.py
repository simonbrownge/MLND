# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:27:08 2017

@author: 108006039
"""
import pandas as pd 
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)
doc_array = count_vector.transform(documents).toarray()
frequency_matrix = pd.DataFrame(doc_array, 
                                columns = count_vector.get_feature_names())
print(frequency_matrix)