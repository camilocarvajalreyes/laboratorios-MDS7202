import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlations(df, method_list):
    corr_list = [df.corr(numeric_only=True, method=method) for method in method_list]
    
    n_corrs = len(corr_list)
    fig, axes = plt.subplots(1, n_corrs, figsize=(5*n_corrs,5))

    if n_corrs == 1:
        axes = [axes]
    # coef_title= ['Pearson', 'Kendall','Spearman']

    for idx, corr in enumerate(corr_list): 
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=axes[idx], cmap='magma', annot=True, fmt='.2f', mask=mask, vmin=0, vmax=1)
        axes[idx].set_title(f'Coeficientes de {method_list[idx].capitalize()}')
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation = 60)

    fig.suptitle('Matrices de correlación', fontsize=16)
    plt.tight_layout()
    return

def boxplot_col_rating_english(df):
    fig=go.Figure()

    fig_buttons = []

    cols = df.select_dtypes(include='number').columns.values
    cols = cols[(cols != 'rating') & (cols != 'english')]
    default_state = cols[0]
    yes_no = ['Yes', 'No']

    for idx, col in enumerate(cols):
        for i, english in enumerate(df['english'].unique()):
            df_plot=df.loc[df['english']==english].copy()


            fig.add_trace(go.Box(x=df_plot["rating"], y=df_plot[col], visible=(col==default_state), name=yes_no[i]))
            fig.update_xaxes(categoryorder='array', categoryarray= ['Negative', 'Mixed', 'Positive', 'Mostly Positive', 'Very Positive'])
            fig.update_layout(legend_title_text='English')
            # fig.for_each_trace(lambda t: t.update(name = {'0':'No', '1': 'Yes'}[t.name]))    

        fig.update_layout(
            boxmode='group',
            title = f'Boxplot de "{default_state}" según rating y disponibilidad en inglés')

        fig_buttons.append(dict(method='update',
                                label=col,
                                args = [{'visible': [col==r for r in cols for i in range(2)]},
                                        {'title':f'Boxplot de "{col}" según rating y disponibilidad en inglés'}]))
    fig.update_layout(
        updatemenus=[
            dict(
                direction='down',
                showactive=True,
                # x=2,
                # y=2,
                buttons=list(fig_buttons)
                ),
            dict(
                buttons=[
                    dict(
                        label="Linear",  
                        method="relayout", 
                        args=[{"yaxis.type": "linear"}]),
                    dict(
                        label="Log", 
                        method="relayout", 
                        args=[{"yaxis.type": "log"}]),
                        ],
                    type='buttons',
                    direction='left',                
                    x=-0.085,
                    y=0.7
                    )],
        annotations=[
            dict(text='y-scale', x=-0.18, xref='paper', y=0.8, yref='paper', align='left', showarrow=False)
        ]
        )
                
    fig.show()



def df_numeric_histograms(df):    
    fig = go.Figure()

    columns = df.select_dtypes(np.number).columns.to_list()
    default_state = columns[0]

    for col in columns:
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                visible=(col==default_state),
                # title_text= f'Distribución de "{col}"'
                )
        )
    fig.update_layout(showlegend=True, title_text=f'Distribución de "{default_state}"') # title of plot

    fig_buttons = []
    visible = np.eye(len(columns)).astype(bool)

    for idx, col in enumerate(columns):
        button_dict = dict(
            label = col,
            method = 'update',
            args = [
                {'visible': visible[idx]},
                # {'title': f'Distribución de "{col}"',
                {'showlegend': True,
                'legend':col}
                ])
        fig_buttons.append(button_dict)

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=list(fig_buttons),
            y=0.9
            ),
            dict(
                buttons=[
                    dict(
                        label="Linear",  
                        method="relayout", 
                        args=[{"xaxis.type": "linear"}]),
                    dict(
                        label="Log", 
                        method="relayout", 
                        args=[{"xaxis.type": "log"}]),
                        ],
                    type='buttons',
                    direction='left',                
                    x=-0.085,
                    y=0.4),
            dict(
                buttons=[
                    dict(
                        label="Linear",  
                        method="relayout", 
                        args=[{"yaxis.type": "linear"}]),
                    dict(
                        label="Log", 
                        method="relayout", 
                        args=[{"yaxis.type": "log"}]),
                        ],
                    type='buttons',
                    direction='left',                
                    x=-0.085,
                    y=0.1)],
        annotations=[
            dict(text='x-scale', x=-0.18, xref='paper', y=0.45, yref='paper', align='left', showarrow=False),
            dict(text='y-scale', x=-0.18, xref='paper', y=0.12, yref='paper', align='left', showarrow=False),
            dict(text='Columna', x=-0.18, xref='paper', y=0.98, yref='paper', align='left', showarrow=False)
        ])
    return fig


def n_best_rating(df, col, n_min_prod=5):
    separate = lambda string: np.array(string.split(';'),dtype=object)

    df.replace({'rating':{'Negative':1, 'Mixed':2, 'Mostly Positive':3, 'Positive':4, 'Very Positive':5}}, inplace=True)
    df = df.astype({'rating': float})
    df.loc[:, col] = df[col].apply(separate)

    df_ratings = df\
                .explode(col)\
                .groupby([col])\
                .agg(
                    rating_promedio=('rating',np.mean),
                    n_juegos=('name','count'),
                    tiempo_de_juego_total=('average_playtime','sum'),
                    total_de_ventas=('estimated_sells','sum'))\
                .sort_values(
                    'rating_promedio',
                    ascending=False)

    df_ratings.columns = ['Rating promedio', 'N° juegos', 'Tiempo de juego total', 'Total de ventas']
    df_ratings = df_ratings.loc[df_ratings['N° juegos'] > n_min_prod].reset_index()

    df = df_ratings.iloc[:100]
    df_n_prod = df_ratings.sort_values('N° juegos', ascending=False).iloc[:100]
    df_playtime_prod = df_ratings.sort_values('Tiempo de juego total', ascending=False).iloc[:100]
    df_sells = df_ratings.sort_values('Total de ventas', ascending=False).iloc[:100]
    df_computations = [df_n_prod, df_playtime_prod, df_sells]

    fig = px.bar(
                df,
                x=col,
                y="N° juegos",
                height=600,
                color='Rating promedio',
                range_color=(1,5),
                title=f'Top "N° juegos" para "{col}"<br>ordenado por "rating"'
            )


    sort_cat_buttons = []
    sort_rat_buttons = []
    for idx, cat in enumerate(['N° juegos', 'Tiempo de juego total', 'Total de ventas']):
        cat_dict = dict(
            label=f"{cat}",
            method="update",
            args=[{"x":[df_computations[idx][col]],
            "y": [df_computations[idx][cat]],
            "marker.color": [df_computations[idx]['Rating promedio']]},
            {'title':f'Top {cat} para "{col}"',
            'yaxis':{'title':cat}}
            # {'yaxis.title':cat},
            ],
            )

        sort_cat_buttons.append(cat_dict)
        rat_dict = dict(
            label=f"{cat} según rating",
            method="update",
            args=[{"x":[df[col]],
            "y": [df[cat]],
            "marker.color": [df['Rating promedio']]},
            {'title':f'Top {cat} para "{col}"<br>ordenado por "rating"',
            'yaxis':{'title':cat}}
            ],
            )
        sort_rat_buttons.append(rat_dict)

    fig.update_layout(
        updatemenus=[
            dict(
                # type="buttons",
                direction="down",
                x=0.55,
                y=1.2,
                showactive=True,
                buttons=list(sort_cat_buttons + sort_rat_buttons
                ),
            ),
            dict(
                direction='right',
                type='buttons',
                x=0.9,
                y=1.2,
                showactive=True,
                buttons=list(
                    [
                        dict(
                            label="Top 100",  
                            method="relayout", 
                            args=[{"xaxis.range": [0-0.5, 100-0.5]}]),
                        dict(
                            label="Top 50",  
                            method="relayout", 
                            args=[{"xaxis.range": [0-0.5, 50-0.5]}]),
                        dict(
                            label="Top 10",  
                            method="relayout", 
                            args=[{"xaxis.range": [0-0.5, 10-0.5]}]),
                        dict(
                            label="Top 5",  
                            method="relayout", 
                            args=[{"xaxis.range": [0-0.5, 5-0.5]}]),        
                    ]
                )
            ),
        ],
    annotations=[
        dict(text='Top N: ', x=0.72, xref='paper', y=1.17, yref='paper', align='left', showarrow=False),
        dict(text='Métrica ordenada por:', x=0.28, xref='paper', y=1.17, yref='paper', align='left', showarrow=False)
    ]
    )
            
    fig.show()    