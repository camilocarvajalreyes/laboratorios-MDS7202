from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import classification_report, r2_score, mean_squared_error
import pandas as pd
import re


class Nothing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X


class CategoriesTokenizer:
    def __init__(self):
        pass

    def __call__(self, doc):
        return doc.split(';')


boc_some_values = CountVectorizer(
    tokenizer = CategoriesTokenizer(),
    max_df = 1.0,
    min_df = 0.05  # hiperparametro a optimizar
    # valores para GridSearch : [5%, 10%, 15%] ???
    )


boc_many_values = CountVectorizer(
    tokenizer = CategoriesTokenizer(),
    max_df = 1.0,
    min_df = 1  # hiperparametro a optimizar
    # valores para GridSearch : [5, 10, 15] ???
    )


def custom_features(dataframe_in):
    df = dataframe_in.copy(deep=True)

    df['month'] = pd.to_datetime(df['release_date']).dt.month
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.to_julian_date())

    df['revenue'] = pd.Series([0 for _ in range(len(dataframe_in))])

    df.loc[df.publisher.str.match('.*microsoft.*', flags=re.IGNORECASE).values, 'revenue'] = 10.260
    df.loc[df.publisher.str.match('.*netease.*', flags=re.IGNORECASE).values, 'revenue'] = 6.668
    df.loc[df.publisher.str.match('.*activision.*', flags=re.IGNORECASE).values, 'revenue'] = 6.388
    df.loc[df.publisher.str.match('.*electronic.*', flags=re.IGNORECASE).values, 'revenue'] = 5.537
    df.loc[df.publisher.str.match('.*bandai.*', flags=re.IGNORECASE).values, 'revenue'] = 3.018
    df.loc[df.publisher.str.match('.*square.*', flags=re.IGNORECASE).values, 'revenue'] = 2.386
    df.loc[df.publisher.str.match('.*nexon.*', flags=re.IGNORECASE).values, 'revenue'] = 2.286
    df.loc[df.publisher.str.match('.*ubisoft.*', flags=re.IGNORECASE).values, 'revenue'] = 1.446
    df.loc[df.publisher.str.match('.*konami.*', flags=re.IGNORECASE).values, 'revenue'] = 1.303
    df.loc[df.publisher.str.match('.*SEGA.*').values, 'revenue'] = 1.153
    df.loc[df.publisher.str.match('.*capcom.*', flags=re.IGNORECASE).values, 'revenue'] = 0.7673
    df.loc[df.publisher.str.match('.*warner.*', flags=re.IGNORECASE).values, 'revenue'] = 0.7324

    return df


preprocessing = ColumnTransformer(
    transformers=[
        ('BoC-plat',boc_some_values,'platforms'),
        ('BoC-cat',boc_some_values,'categories'),
        ('BoC-genres',boc_some_values,'genres'),
        ('BoC-tags',boc_some_values,'tags'),

        ('BoC-dev',boc_many_values,'developer'),
        ('BoC-pub',boc_many_values,'publisher'),

        ('OneHotEncoder',OneHotEncoder(handle_unknown='ignore'),['month']),
        ('MinMaxScaler',MinMaxScaler(),['required_age','price','release_date']),
        ('BoxCox',PowerTransformer(method='yeo-johnson'),['achievements','average_playtime','revenue']),
        # ('unchanged',Nothing(),['english'])
])


def make_pipeline(clf,column_transformer,perc=95):
    pipe = Pipeline([
        ('Pre-procesamiento',column_transformer),
        ("selector", SelectPercentile(f_classif, percentile=perc)),
        (type(clf).__name__,clf),
    ])
    return pipe


def train_and_evaluate_clf(clf,X_train,y_train,X_eval,y_eval,perc=95):
    pipe = make_pipeline(clf,preprocessing,perc)
    print("Resultados clasificación {}".format(type(clf).__name__))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_eval)
    print(classification_report(y_eval,y_pred))


def train_and_evaluate_reg(clf,X_train,y_train,X_eval,y_eval,perc=95):
    pipe = make_pipeline(clf,preprocessing,perc)
    print("Resultados regresión {}".format(type(clf).__name__))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_eval)
    print("Error cuadrático medio = {}".format(mean_squared_error(y_eval,y_pred)))
    print("Score R2 = {}".format(r2_score(y_eval,y_pred)))


def custom_features(dataframe_in):
    df = dataframe_in.copy(deep=True)

    df['month'] = pd.to_datetime(df['release_date']).dt.month
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.to_julian_date())
    return df
