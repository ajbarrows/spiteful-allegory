import os
import Bio
import pandas as pd
import numpy as np
import requests
import xmltodict
import nltk
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed, cpu_count

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
    abs = df['abstract'].str.cat()
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

    # turn abstract into sentences
    sentences = nltk.sent_tokenize(abstract)

    # drop first and last sentence (!!)
    sentences = sentences[1:-1]

    corp = set(corpus.values())

    detected = []
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

            # keywords = []
            # for word in overlap:
            #     for k, v in corpus.items():
            #         if v == word:
            #             keywords.append(k)
            
            # detected[sentence] = keywords
            detected.append(sentence)

    return {study_id: "".join(detected)}

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
    detected['det_sentences'] = out.values()

    df = (df
          .set_index('study_id')
          .join(detected.set_index('study_id'))
          .reset_index()
          )

    return df