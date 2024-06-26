import os
import Bio
import pandas as pd
import numpy as np
import requests
import xmltodict
from bertopic import BERTopic

import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed, cpu_count

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset

from work.pipelines.word_helpers.nodes import *

def establish_download_directory(fpath="data/01_raw/neuro-text/"):

    out_dir = os.path.abspath(fpath)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir

def download_neurodata(out_dir, dataset: str):

    if dataset == "neurosynth":
        files = fetch_neurosynth(
            data_dir=out_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
    elif dataset == "neuroquery": 
        files = fetch_neuroquery(
            data_dir=out_dir,
            version="1",
            overwrite=False,
            source="combined",
            vocab="neuroquery6308",
            type="tfidf",
        )

    ndb = files[0]

    return ndb


def convert_dset(ndb):
     n_dset = convert_neurosynth_to_dataset(
        coordinates_file=ndb["coordinates"],
        metadata_file=ndb["metadata"],
        annotations_files=ndb["features"],
        )
     
     return n_dset


def download_neuro_abstracts(n_dset, email="example@example.edu"):
    n_dset = download_abstracts(n_dset, email)
    return n_dset


def _join_metadata(texts, metadata):

    idx = ['id', 'study_id', 'contrast_id']
    return (texts
            .set_index(idx)
            .join(metadata.set_index(idx))
            .reset_index()
        )

def save_neurodata(n_dset, dataset:str, out_dir:str):
    '''Save full dataset but return abstracts.'''

    # save full dataset
    fname = f"{dataset}_dataset_with_abstracts.pkl.gz"
    n_dset.save(os.path.join(out_dir, fname))

    # assemble full article info
    out = _join_metadata(n_dset.texts, n_dset.metadata)

    return out

def drop_duplicated_neuroquery(ns, nq):
    '''
    Find any articles that are in both databases.
    Drop them from neuroquery (less complete metadata)
    '''
    
    nq = nq[~nq['study_id'].isin(ns['study_id'])]
    return ns, nq

def get_pubmed_citations(nq):
    '''Query NIH API. Likely request-limited. Use caution.'''
    
    service_root= 'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi'
    texts = nq['study_id']

    citations = []

    for text in texts:

        # compose query
        text = 'PMC' + text
        req = f'{service_root}?id={text}'

        # parse and hault out citation
        resp = requests.get(req)
        pub_meta = xmltodict.parse(resp.content)

        try:
            citation = pub_meta['OA']['records']['record']['@citation']
        except KeyError:
            citation = np.NaN

        citations.append(citation)

    nq_with_citations = pd.concat([
        nq, 
        pd.Series(citations, name='citation')], 
        axis=1)

    return nq_with_citations

def extract_dates_from_citations(df):

    pattern='[.|,|;]'

    expanded = df['citation'].str.split(pattern, regex=True, expand=True)
    expanded = expanded.astype('str')

    for c in expanded.columns:
            
        expanded[c] = pd.to_datetime(
            expanded[c].str.strip(), 
            format="%Y %b %d",
            errors='coerce')
        
    contracted = expanded.astype('str').sum(axis=1)
    contracted = pd.to_datetime(contracted.str.replace('NaT', ''))

    df['year'] = pd.DatetimeIndex(contracted).year

    return df

def combine_texts(ns: pd.DataFrame, nq: pd.DataFrame):
    
    ns['source'] = 'neurosynth'
    nq['source'] = 'neuroquery'

    return pd.concat([ns, nq])

def rankcount_from_abstracts(df: pd.DataFrame):

    # turn abstracts into single string, parse
    # abs = df['abstract'].str.cat()
    abs = df['det_sentences'].str.cat()
    parsed = parse_text(abs)
    rankcount = generic_sizerank_df(parsed)

    return parsed, rankcount


def _filter_pos(df, pos_to_keep: list, simple_verbs: list):

    df = df.drop(columns='rank')
    df = df[df['pos'].isin(pos_to_keep)]
    df = df[~df['ngram'].isin(simple_verbs)]

    df = (df
     .reset_index(drop=True)
     .reset_index(names='rank')
    )
    # reindex rank
    df['rank'] = df['rank'] + 1

    return df


def add_nltk_pos(df: pd.DataFrame, pos_to_keep: list, simple_verbs: list) -> pd.DataFrame:

    tags = nltk.pos_tag(df['ngram'], tagset='universal')
    pos = []
    for t in tags:
        pos.append(t[1])

    df['pos'] = pos

    df = _filter_pos(df, pos_to_keep, simple_verbs)

    return df


def select_random_abstracts(df, n_random=50, random_state=42):
    rnd = df.sample(n=n_random, random_state=random_state)
    return rnd[['study_id', 'abstract']]


def load_methods_corpus(text):
    ps = PorterStemmer()
    corpus = text.split('\n')
    stems = [ps.stem(word) for word in corpus]
   
    return dict(zip(corpus, stems))

def load_skip_words(skip_words: list):
    ps = PorterStemmer()
    skip = set(ps.stem(s) for s in skip_words)

    return skip

def text_to_dict(df):
    return dict(zip(df['study_id'], df['abstract']))

def detect_corpus_in_sentences(text: tuple, corpus: dict, skip_words: set):

    study_id = text[0]
    abstract = text[1]

    ps = PorterStemmer()

    try:
        # turn abstract into sentences
        sentences = nltk.sent_tokenize(abstract)
    except:
        return {study_id: 'invalid_abstract'}
      

    # drop first and last sentence (!!)
    sentences = sentences[1:-1]

    corp = set(corpus.values())

    detected = []
    keywords = []
    for sentence in sentences:

        # turn sentences into tokens
        parsed = parse_text(sentence)
        tokens = parsed.split()

        # turn tokens into stems
        stems = set([ps.stem(t) for t in tokens])
        
        # stop if a break word comes up
        if any(s in skip_words for s in stems):
            break
        
        # detect corpus words in sentence
        overlap = stems.intersection(corp)
        
        if len(overlap) > 0:
            for word in overlap:
                for k, v in corpus.items():
                    if v == word:
                        keywords.append(k)
            
            detected.append(sentence)
    
    # reglue
    detected = " ".join(detected)
    out = {study_id: [set(keywords), detected]}

    return out

def run_detect_parallel(abstract_dict, methods_corpus, skip_words):

    n_cpus = cpu_count()

    r = Parallel(n_jobs=n_cpus)(
        delayed(detect_corpus_in_sentences)(item, methods_corpus, skip_words) for item in abstract_dict.items()
    )

    return r

def gather_detected_output(df: pd.DataFrame, res: list):
    
    out = {k: v for d in res for k, v in d.items()}

    detected = pd.DataFrame()

    detected['study_id'] = out.keys()
    detected['keywords'] = [', '.join(v[0]) for v in out.values()]
    detected['det_sentences'] = [v[1] for v in out.values()]


    df = (df
          .set_index('study_id')
          .join(detected.set_index('study_id'))
          .reset_index()
          )

    return df

def _process_text(input_text):

    # 1-grams
    tokens = word_tokenize(input_text)

    # remove stop words
    stop_words = set(stopwords.words('english'))

    # condense words into lemma
    lem = WordNetLemmatizer()

    # execute
    text = [lem.lemmatize(t.lower()) for t in tokens if t not in stop_words]

    # recycle
    out = " ".join(text)

    return out


def add_cleaned_text_column(df: pd.DataFrame, input_col = 'det_sentences'):
    
    df = df[df[input_col] != '']
    df['cleaned'] = df[input_col].apply(_process_text)
    docs = df['cleaned'].to_list()

    return df, docs

def fit_BERTopic_model(docs: list):

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    return topic_model

def fit_lda_model(docs: list):

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(n_components=10)
    lda_fit= lda.fit_transform(X)

    return lda, vectorizer

def extract_BERTopic_topics(mod, timepoint=None):

    topics = mod.get_topics()
    frequencies = mod.get_topic_freq()
    topic_idx = list(topics)[1:]
    first_n_topics = {k:topics[k] for k in topic_idx}

    bert_topics = pd.DataFrame()

    for k, v in first_n_topics.items():
        tmp = pd.DataFrame()
        topic = 'Topic ' + str(k)
        term = [t[0] for t in v]
        weight = [t[1] for t in v]
        frequency = frequencies['Count'].iloc[k]


        tmp['term'] = term
        tmp['weight'] = weight
        tmp['topic'] = topic
        tmp['frequency'] = frequency

        bert_topics = pd.concat([bert_topics, tmp])

    bert_topics['model'] = 'BERTopic'

    if timepoint:
        bert_topics['timepoint'] = timepoint

    return bert_topics

def extract_LDA_topics(mod, vectorizer, timepoint=None):

    lda_topics = pd.DataFrame()
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(mod.components_):
        top_features_ind = topic.argsort()[::-1]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        tmp = pd.DataFrame()
        topic = 'Topic ' + str(topic_idx)

        tmp['term'] = top_features
        tmp['weight'] = weights
        tmp['topic'] = topic

        lda_topics = pd.concat([lda_topics, tmp])

    lda_topics['model'] = 'LDA'

    if timepoint:
        lda_topics['timepoint'] = timepoint
        
    return lda_topics

def join_output(*dfs):
    return pd.concat(dfs)

def fit_models_by_group(df: pd.DataFrame, year_map: dict):

    df['year_group'] = df['year']
    df['year_group'] = df['year_group'].replace(year_map)

    lda_models = {}
    bert_models = {}
    output = pd.DataFrame()
    grouped = df.groupby('year_group')
    for group_name, df in grouped:
        
        df, docs = add_cleaned_text_column(df)

        topic_model = fit_BERTopic_model(docs)
        lda, vectorizer = fit_lda_model(docs)

        bert_topics = extract_BERTopic_topics(topic_model, timepoint=group_name)
        lda_topics = extract_LDA_topics(lda, vectorizer, timepoint=group_name)

        tmp = join_output(bert_topics, lda_topics)

        output = pd.concat([output, tmp])

        lda_models[group_name] = (lda, vectorizer)
        bert_models[group_name] = topic_model

    return output, lda_models, bert_models