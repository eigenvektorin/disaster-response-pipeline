import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, precision_score,\
recall_score,accuracy_score,  f1_score,  make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from CharacterCounter import CharacterCounter
from CapitalLetterCounter import CapitalLetterCounter

def load_data(database_filepath):
    """  Loads dataset from sql database and splits in onto into X and Y variables
     Args:
        database_filepath - Python str object - path to the SQL database

    OUTPUT:
        X - A pd series containing the messages
        Y - A pd dataframe containing the category columns
        Y.columns - A list that holds all the category names for the messages
        """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns

def tokenize(text):
    """ Tokenization function that processes the text data """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
       Args:
           None
       Returns:
           cv - A grid-search pipeline used to train the model and find the best parameters
       """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('char_count', CapitalLetterCounter())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # specify parameters for grid search
    parameters = {'clf__estimator__n_estimators': [150, 200, 250],
                  #'clf__estimator__max_depth': [5, 10, 12]
                  }

    # create grid search object
    # cv = GridSearchCV(pipeline, param_grid = parameters)
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """ Evaluates model on the test data and prints the result
    Args:
        model  = trained classifier model
        X_test = the test data
        Y_test = true values to compare with prediction on unseen test cases
        category_names = column names of Y_test data
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    metrics_list_all = []
    for i in range(y_test.shape[1]):
        accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        precision = precision_score(y_test.iloc[:, i], y_pred[:, i])
        recall = recall_score(y_test.iloc[:, i], y_pred[:, i])
        f_1 = f1_score(y_test.iloc[:, i], y_pred[:, i])
        metrics_list = [accuracy, precision, recall, f_1]
        metrics_list_all.append(metrics_list)

    metrics_df = pd.DataFrame(metrics_list_all, index=category_names, columns=
    ["Accuracy", "Precision", "Recall", "F_1"])
    print(metrics_df)


def save_model(model, model_filepath):
    """ Exports model as pickle file"""
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)    #=Y.columns

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()