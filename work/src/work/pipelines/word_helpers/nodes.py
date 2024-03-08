"""
This is a boilerplate pipeline 'word_helpers'
generated using Kedro 0.18.14
"""

import numpy as np
import pandas as pd
import re
from collections import Counter


def get_avg_happiness(text, map, filter=(4,6), rtn_datasets=False):
    '''
    Return the average/weighted average happiness for a particular
    lexical lens. Optinally return filtered dataset with happiness
    scores for the supplied words also in `map`.
    '''

    if filter:
        lens_min = filter[0]
        lens_max = filter[1]

        map = map[(map['happiness'] <= lens_min) | (map['happiness'] >= lens_max)]

    joined = (text
              .set_index('word')
              .join(map.set_index('word'))
              .dropna())
    
    avg = joined['happiness'].mean()
    weighted_avg = np.average(a=joined['happiness'], weights=joined['count'])

    rtn = {
        'h_avg': avg,
        'weighted_h_avg': weighted_avg
    }

    ds = {
        'joined': joined,
        'map': map
    }

    if rtn_datasets:
        return rtn, ds
    else:
        return rtn

    
def split_narrative_ts(ts: list, first_middle_last='first', prop=1, prop2=None, return_counts=True) -> dict:
    '''Return the first/last portion (prop) of a list.'''

    N = len(ts)

    if first_middle_last == 'first':
        min_idx = 0
        max_idx = int(prop * N)
    elif first_middle_last == 'middle':
        min_idx = int(prop * N)
        max_idx = int(prop2 * N)
    elif first_middle_last == 'last':
        min_idx = int(prop * N)
        max_idx = N-1

    frac = ts[min_idx:max_idx]

    if return_counts:
        counts = Counter(frac).most_common()
        counts = dict(counts)
    
    rtn = {
        'narrative_ts_frac': frac,
        'counts': counts
    }
    
    return rtn

def happines_from_count_dict(counts: dict, ref, filter=(4,6)):  

    df = pd.DataFrame.from_dict(
        counts['counts'], 
        orient='index', 
        columns=['count'])
    df = df.reset_index(names=['word'])

    avg = get_avg_happiness(df, ref, filter)

    return avg['weighted_h_avg']


def happiness_from_narrative_ts(narrative_ts: list, map: pd.DataFrame, coerce_to_lower=True) -> pd.DataFrame:
    '''Join with `map` to produce happiness time series.'''

    idx = 'word'
    df = pd.DataFrame(narrative_ts, columns=['word'])

    if coerce_to_lower:
        df['word'] = [w.lower() for w in df['word']]

    df = (df
    .join(map.set_index(idx), on=idx)
    # .dropna(subset='happiness')
    # .reset_index(drop=True)
    .reset_index(names='word_order')
    )

    return df

def filter_delta_h(df: pd.DataFrame, lens, set_point=5):
    
    if type(lens) is tuple:
        lens_min = lens[0]
        lens_max = lens[1]
    else:
        lens_min = set_point - lens/2
        lens_max = set_point + lens/2


    return df[(df['happiness'] <= lens_min) | (df['happiness'] >= lens_max)]

def make_size_rank_dist(text, fname=None, limit_top_n=None, write_file=True, 
                        filepath = '../data/03_primary/gutenberg_size_rank/') -> list:

    onegrams = text.split()
    counts = Counter(onegrams)

    # sort
    if limit_top_n:
        counts = counts.most_common(limit_top_n)
    else:
        counts = counts.most_common()

    if write_file:
        fpath = filepath + fname
        with open(fpath, 'w') as file:
            for k, v in counts:
                file.write("{}\t{}\n".format(k, v))

    return counts

def make_dataframe_from_sizerank(sizerank) -> pd.DataFrame:

    df = pd.DataFrame()

    for k, v in sizerank.items():
        tmp = pd.DataFrame(v, columns=['ngram', 'count'])
        tmp.insert(0, 'source', k)
        tmp = tmp.reset_index(names = ['rank'])

        df = pd.concat([df, tmp])
    
    return df

def parse_text(t):
    '''Return text corpus separated into 1-grams.'''

    # Frankenstein thing
    t = re.sub(r'D--n', r'Damn', t)

    # remove dashes with a period following
    t = re.sub(r'—.', r'', t)

    # separate punctuation
    t = re.sub(r'(?=[.,!?:;])', r' ', t)

    # deal with brackets and such
    t = re.sub(r'(?<=[\(\[])', r' ', t)
    t = re.sub(r'(?=[\)\]])', r' ', t)

    # deal with quotes
    t = re.sub(r'(?=[”])', r' ', t)
    t = re.sub(r'(?<=[“\'])', r' ', t)
    t = re.sub(r'\'s', ' \'s', t) # possession

    # dash madness
    t = re.sub(r'----', r' --- ', t)
    t = re.sub(r'--', r' --- ', t)
    t = re.sub(r';-', r' --- ', t)
    t = re.sub(r'—', r' --- ', t)

    ## TODO make this handle "Mr" and "Mrs" with no period
    # handle specific salutations
    t = re.sub(r'Mr .', r'Mr.', t)
    t = re.sub(r'Mrs .', r'Mrs.', t)
    t = re.sub(r'Dr .', r'Dr.', t)

    # remove underscores used for emphasis
    t = re.sub(r'_', r'', t)

    # remove string control
    t = re.sub(r'[\n\r\t]', ' ' , t)

    # remove additional whitespaces
    t = re.sub(r'\s+', ' ', t)

    return t