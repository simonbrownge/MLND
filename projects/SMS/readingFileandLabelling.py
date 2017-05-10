# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:37:42 2017

@author: 108006039
"""

import pandas as pd 
#from "https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection"
df = pd.read_table('SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
print df.head()