import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """   Loads and merges datasets.
    Args:
    messages_filepath - str object - path to messages.csv
    categories_filepath - str object - path to categories.csv
    Returns:
    df - Pandas DataFrame by merging the messages and categories datasets on the common id  """
    messages = pd.read_csv(messages_filepath,encoding='utf-8')
    categories = pd.read_csv(categories_filepath, encoding='utf-8')
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """ Splits categories into separate category columns and converts values to 0 and 1; remove duplicates.
    Args:
    df - Pandas DataFrame object - the output from last step
    Returns:
    df - celaned Pandas DataFrame object
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories[:1]
    category_colnames = np.array(row.apply(lambda x: x.str[0:-2])).flatten()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    categories = categories.replace(2, 1)
    df =df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """ Saves the clean dataset into an sqlite database.
    Args:
    df - Pandas DataFrame object - the cleaned dataset from last step
    database_filename - Python str object - name of SQL database file that hold the data
    Returns:
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data('./disaster_messages.csv', './disaster_categories.csv')

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()