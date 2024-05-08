import os
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

def get_top_n_topics(df, MAX_TOPICS = 10):
    topics = pd.unique(df['topic'])
    return(topics[:MAX_TOPICS])

def split_models(df):

    bert = df[df['model'] == 'BERTopic']
    lda =  df[df['model'] == 'LDA']

    return bert, lda

def filter_data(bert, lda, topic):

    bert_top = bert[bert['topic'] == topic].sort_values('weight')
    lda_top = lda[lda['topic'] == topic].iloc[:10].sort_values('weight')

    return bert_top, lda_top

def filter_timepoint(df, timepoint):
    
    bert = df[(df['model'] == 'BERTopic') & (df['timepoint'] == timepoint)]
    lda = df[(df['model'] == 'LDA') & (df['timepoint'] == timepoint)]

    return bert, lda

def produce_top_n_topics(bert, lda, top_n_topics, fpath='./data/08_reporting/project/top_n_topics.pdf'):

    fig = plt.figure(figsize=(14, 12))
    subfigs = fig.subfigures(5, 2, hspace=0, wspace=0.1)

    for subfig, topic, in zip(subfigs.flat, top_n_topics):

        bert_top, lda_top = filter_data(bert, lda, topic)

        axs = subfig.subplot_mosaic([['top', 'top'], ['bottom_left', 'bottom_right']], gridspec_kw={'height_ratios': [0.05, 10]})

        ax = axs['top']
        ax.text(0.5, -0.1, topic, ha='center', va='center')
        ax.set_axis_off()

        ax = axs["bottom_left"]
        ax.barh(lda_top['term'], lda_top['weight'], color='skyblue')
        ax.set_xticks([])
        ax.set_xlabel('LDA')

        ax = axs["bottom_right"]
        ax.barh(bert_top['term'], bert_top['weight'] * -1, color='mediumorchid')
        ax.yaxis.tick_right()
        ax.set_xticks([])
        ax.set_xlabel('BERTopic')

    plt.savefig(fpath, bbox_inches='tight')


def produce_topics_by_year(df, topic='Topic 0', fpath='./data/08_reporting/project/topics_by_timepoint.pdf'):

    fig = plt.figure(figsize=(14, 6))
    subfigs = fig.subfigures(2, 2, hspace=0, wspace=0.1)

    timepoints = pd.unique(df['timepoint'])

    for subfig, timepoint, in zip(subfigs.flat, timepoints):

        bert, lda = filter_timepoint(df, timepoint)
        bert_top, lda_top = filter_data(bert, lda, topic)

        axs = subfig.subplot_mosaic([['top', 'top'], ['bottom_left', 'bottom_right']], gridspec_kw={'height_ratios': [0.05, 10]})

        ax = axs['top']
        title = "Years " + timepoint
        ax.text(0.5, -0.1, title, ha='center', va='center')
        ax.set_axis_off()

        ax = axs["bottom_left"]
        ax.barh(lda_top['term'], lda_top['weight'], color='skyblue')
        ax.set_xticks([])
        ax.set_xlabel('LDA')

        ax = axs["bottom_right"]
        ax.barh(bert_top['term'], bert_top['weight'] * -1, color='mediumorchid')
        ax.yaxis.tick_right()
        ax.set_xticks([])
        ax.set_xlabel('BERTopic')

    plt.savefig(fpath, bbox_inches='tight')

def produce_appendix(df, top_n_topics, fpath = './data/08_reporting/project/appendix/'):

    timepoints = pd.unique(df['timepoint'])

    
    for timepoint in timepoints:
        fname = timepoint + '.pdf'
        fdir = os.path.join(fpath, fname)

        bert, lda = filter_timepoint(df, timepoint)
        produce_top_n_topics(bert, lda, top_n_topics, fpath=fdir)
        

