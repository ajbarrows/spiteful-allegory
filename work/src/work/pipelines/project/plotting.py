import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use('science')

plt.rcParams["xtick.minor.visible"] =  False
plt.rcParams["ytick.minor.visible"] =  False
plt.rcParams['legend.frameon'] = True

def _count_publications(df: pd.DataFrame):

    df_agg = df.groupby('year').count()['study_id']
    return df_agg.reset_index()

def _decrease_plot_space(n_points):
    '''Make labels flip-flop around.'''
    space_array = np.linspace(10, 100, n_points)
    posneg_array = np.resize([-1, 1], n_points)

    return space_array * posneg_array

def _make_annotation_dict(df, first_n, n_words_sampled, drop_n_prop=0):
    df = df.drop(columns='pos')
    df.columns = ['x', 'text', 'y']

    n_last = int(len(df) - len(df) * drop_n_prop)

    df_first_n = df.iloc[0:first_n] # grab the first_n terms
    df_last = df.iloc[[-1]] # grab the lowest-ranked term

    # grab a sample of words between two above, weighted by their frequency
    df_select = df.iloc[first_n+1:n_last]
    df_select= df_select.sample(n_words_sampled, weights=df['y'])

    df = pd.concat([df_first_n, df_select, df_last])

    df['x'] = np.log10(df['x'])
    df['y'] = np.log10(df['y'])

    df['ay'] = _decrease_plot_space(len(df))

    return df.to_dict('records')


def make_rankcount_plot(df, first_n=10, n_words_sampled=10,
                        fpath='./data/08_reporting/project/rankcount.pdf'):

    annotations = _make_annotation_dict(df, first_n, n_words_sampled)

    fig1 = go.Scatter(
        x=df['rank'], 
        y=df['count'], 
        mode='markers'
        )

    # embedded plot
    fig2 = go.Histogram(
        x=df['pos'],
        xaxis='x2',
        yaxis='y2')

    data = [fig1, fig2]
    layout = go.Layout(
        yaxis=dict(
            title = dict(text='Number of Appearances'), 
            type='log',
            showline=True,
            linecolor='grey',
            linewidth=1
            ),
        xaxis=dict(
            title=dict(text='Term Rank'), 
            type='log',
            showline=True,
            linecolor='grey',
            linewidth=1
            ),

        # embedded plot position
        xaxis2=dict(
            domain=[0.1, 0.3],
            anchor='y2'
        ),

        yaxis2=dict(
            domain=[0.1, 0.4],
            anchor='x2'
        ),
        annotations=annotations,
        font_family="Computer Modern",
        showlegend=False,
        plot_bgcolor= "rgba(0, 0, 0, 0)",
        paper_bgcolor= "rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400
        # title=dict(text='Rank-count distribution for content words')
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_image(fpath)



# def plot_pubcounts(df, title="Neurosynth Publication Counts", 
#                    fpath='./data/08_reporting/project/ns_pubcount.pdf'):

#     pubcounts = _count_publications(df)

#     fig = px.line(pubcounts, x='year', y='study_id')
#     fig.update_layout(
#         title=title,
#         xaxis_title='Year',
#         yaxis_title='Publications',
#         font_family="Computer Modern"
#     )
#     fig.write_image(fpath, width=700, height=500)

def plot_pubcounts(df, fpath='./data/08_reporting/project/ns_pubcount.pdf'):

    pubcounts = _count_publications(df)
    fig, ax = plt.subplots(figsize=(4, 2))

    ax.plot(pubcounts['year'], pubcounts['study_id'])
    ax.set_ylabel('Publications')
    ax.set_xlabel('Year')

    plt.savefig(fpath)


