"""Some text basic exploration functions"""
from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

from nltk import ngrams
from nltk.corpus import stopwords
from wordcloud import WordCloud


def most_common(df:pd.DataFrame,column:str,n_grams:int=1,limit:int=10,ignore:list=[]) -> pd.Series:
    """
    Takes a df and a column (name of it, as a string), it returns a Pandas Series with n-grams with their respective frequency

    Arguments

        df: pd.DataFrame
            dataframe containing text to analyse

        column: str
            target column name (from df) with text content
        
        n_grams: int, default 1
            whether to  consider unigrams (1, i.e., single words), two grams, three grams, etc

        limit: int, default 10
            limits the number of most-common n_grams to return
        
        ignore: list, default []
            list of words to ignore

    Returns

        tokens: pd.Series
            pandas Series with n_gram to frequency for the given dataframe and column
    
    """
    if n_grams == 1:
        tokens =  pd.Series(' '.join(df[column]).lower().split()).value_counts()
        tokens = tokens.drop(ignore,errors='ignore')
        tokens = tokens[:limit]
    else:
        ngram_counts = Counter(ngrams(' '.join(df[column]).lower().split(), n_grams))
        comunes_2g = ngram_counts.most_common(limit)
        tokens = pd.Series([tup[1] for tup in comunes_2g],index=[' '.join(tup[0]) for tup in comunes_2g])
    
    return tokens

def plot_token_frequency(data, n_grams_list, title):
    serie_tokens = most_common(
        data,
        'short_description',
        ignore=stopwords.words('english'),
        n_grams=n_grams_list[0],
        limit=100)

    fig = px.bar(
                x=serie_tokens,
                y=serie_tokens.index,
                height=600,
                # color='Rating promedio',
                range_color=(1,5),
                title=title + f' - {n_grams_list[0]} n gramas',
                orientation='h'
                )

    fig_buttons = []
    for idx, n_grams in enumerate(n_grams_list):
        serie_tokens = most_common(data,'short_description',ignore=stopwords.words('english'), n_grams=n_grams, limit=100)
        cat_dict = dict(
            label=f"{n_grams}",
            method="update",
            args=[{"x":[serie_tokens],
            "y": [serie_tokens.index]},
            {'title':title + f' - {n_grams} n gramas',
            'yaxis':{'title':'Tokens'},
            'xaxis':{'title':'Frecuencia'}}
            ],
            ) 
        fig_buttons.append(cat_dict)

    fig.update_layout(
        updatemenus=[
            dict(
                direction="down",
                y=0.8,
                x=-0.12,
                showactive=True,
                buttons=list(fig_buttons
                ),
            ),
            dict(
                direction='down',
                type='buttons',
                # x=-0.1,
                y=0.4,
                showactive=True,
                buttons=list(
                    [
                        dict(
                            label="Top 100",  
                            method="relayout", 
                            args=[{"yaxis.range": [0-0.5, 100-0.5]}]),
                        dict(
                            label="Top 50",  
                            method="relayout", 
                            args=[{"yaxis.range": [0-0.5, 50-0.5]}]),
                        dict(
                            label="Top 10",  
                            method="relayout", 
                            args=[{"yaxis.range": [0-0.5, 10-0.5]}]),
                        dict(
                            label="Top 5",  
                            method="relayout", 
                            args=[{"yaxis.range": [0-0.5, 5-0.5]}]),        
                    ]
                )
            )
        ],
        annotations=[
        dict(text='Top N: ', x=-0.155, xref='paper', y=0.44, yref='paper', align='left', showarrow=False),
        dict(text='N gramas:', x=-0.155, xref='paper', y=0.85, yref='paper', align='left', showarrow=False)
    ]
    )

    fig.show()
    return

def wordcloud_from_column(df:pd.DataFrame,column:str,maxfont:int=40,ignore:list=[], ax=None):
    """
    It generates a wordcloud visualisation from the dataframe column (values must be strings)
    source: https://amueller.github.io/word_cloud/auto_examples/simple.html

    Arguments

        df: pd.DataFrame
            dataframe containing text to analyse

        column: str
            target column name (from df) with text content
        
        maxfont: int, default 40
            maximum font to use in visualisation, if set to None it will be generated from frequencies
        
        ignore: list, default []
            list of words to ignore
    
    """
    if ax is None:
        ax = plt.gca()
    # to do example coloured by group: https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html

    text = ' '.join(df[column]).lower()
    
    for word in ignore:
        text = text.replace(' '+ word + ' ',' ')
    # tokens =  pd.Series(' '.join(df[column]).lower().split())

    wordcloud = WordCloud(max_font_size=maxfont).generate(text)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
