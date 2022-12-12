from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import classification_report
import pandas as pd


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
    return df


preprocessisng = ColumnTransformer(
    transformers=[
        ('BoC-plat',boc_some_values,'platforms'),
        ('BoC-cat',boc_some_values,'categories'),
        ('BoC-genres',boc_some_values,'genres'),
        ('BoC-tags',boc_some_values,'tags'),

        ('BoC-dev',boc_many_values,'developer'),
        ('BoC-pub',boc_many_values,'publisher'),

        ('OneHotEncoder',OneHotEncoder(handle_unknown='ignore'),['month']),
        # ('StandardScaler',StandardScaler(), ['...']),
        ('MinMaxScaler',MinMaxScaler(),['required_age','price','release_date']),
        ('BoxCox',PowerTransformer(method='yeo-johnson'),['achievements','average_playtime']),
        # ('unchanged',None,['english'])  # chequear como no hacer nada
])


def make_pipeline(clf,column_transformer,perc=95):
    pipe = Pipeline([
        ('Pre-procesamiento',column_transformer),
        ("selector", SelectPercentile(f_classif, percentile=perc)),
        (type(clf).__name__,clf),
    ])
    return pipe


def train_and_evaluate(clf,X_train,y_train,X_eval,y_eval,perc=95):
    pipe = make_pipeline(clf,preprocessisng,perc)
    print("Resultados clasificación {}".format(type(clf).__name__))
    pipe.fit(X_train, y_train)
    y_svm = pipe.predict(X_eval)
    print(classification_report(y_eval,y_svm))


def custom_features(dataframe_in):
    df = dataframe_in.copy(deep=True)

    df['month'] = pd.to_datetime(df['release_date']).dt.month
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.to_julian_date())
    return df
