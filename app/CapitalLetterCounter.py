import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk import word_tokenize

class CapitalLetterCounter(BaseEstimator, TransformerMixin):
    '''
    Customized transformer for counting capital letters occurances in text.
    '''
    # Adding 'activate' parameter to activate the transformer or not:
    def __init__(self, activate = True):
        self.activate = activate

    # Defining fit method:
    def fit(self, X, y = None):
        return self

    # Defining transform method:
    def transform(self, X):
        '''
        It recieves an array of messages and counts the number of capital
        letters for each message.
        Input:
        X: array of text messages
        Output:
        cap_count_arr: array with the number of capital letters for each message
        '''
        # If activate parameter is set to True:
        if self.activate:
            # Creating empty list:
            cap_count_list = list()
            # Verifying each character to see whether it's a capital letter or not:
            for i in range (len(X)):
                cap_count = 0
                msg = X[i]
                for j in range(len(msg)):
                    if msg[j].isupper():
                        cap_count += 1
                cap_count_list.append(cap_count)
            # Transforming list into array:
            cap_count_arr = np.array(cap_count_list)
            cap_count_arr = cap_count_arr.reshape((len(cap_count_arr),1))
            return cap_count_arr

        # If activate parameter is set to False:
        else:
            pass