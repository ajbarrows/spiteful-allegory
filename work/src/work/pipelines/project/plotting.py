import plotly.express as px

def _count_publications(df: pd.DataFrame):

    df_agg = df.groupby('year').count()['study_id']
    return df_agg.reset_index()


def plot_pubcounts(df, title="Neurosynth Publication Counts", 
                   fpath='../data/08_reporting/project/ns_pubcount.pdf'):

    pubcounts = _count_publications(df)

    fig = px.line(pubcounts, x='year', y='study_id')
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Publications'
    )
    fig.write_image(fpath, width=700, height=500)
