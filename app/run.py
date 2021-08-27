import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, sunburst
import joblib
from sqlalchemy import create_engine
from CapitalLetterCounter import CapitalLetterCounter

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    df_clean = df.drop(['original'], axis=1)
    # TODO: Below is an example - modify to extract data for your own visuals
    # Count messages by genre
    genre_counts = df.genre.value_counts()
    genre_percent = 100*genre_counts/genre_counts.sum()
    genre_names = list(genre_counts.index)

    # Count messages by category
    df_cat = df_clean.drop(columns=['id', 'message', 'genre'])
    cat_count = df_cat.sum().sort_values(ascending=False)
    category_names = cat_count.index
    count = pd.DataFrame(cat_count, columns=['Counts'])
    count.reset_index(level=0, inplace=True)
    count.columns = ['Category', 'Counts']
    top_20 = count.replace(count.groupby('Category').sum().sort_values('Counts', ascending=False).index[20:],
                           'other').groupby('Category').sum()
    top_20.reset_index(level=0, inplace=True)
    top_20.columns = ['Category', 'Counts']

    df_agg = \
    df_clean.melt(id_vars=['message', 'genre'], var_name='category', value_name='count').groupby(['genre', 'category'])[
        'count'].sum().reset_index().query('count > 0')
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [
                {
                    "type": "pie",
                    #"hole": 0.6,
                    "name": "Genre",
                    #"pull": 0,
                    "domain": {
                        "x": genre_percent,
                        "y": genre_names
                    },
                    "marker": {
                        "colors": [
                            "#C38D9E",
                            "#E8A87C",
                            "41B3A3"
                        ]
                    },
                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": genre_names,
                    "values": genre_counts
                }
            ],
            "layout": {
                "title": "Distribution of Messages by Genre"
            }
        },
        {
            'data': [
                Bar(
                    x=top_20.Category,
                    y=top_20.Counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()