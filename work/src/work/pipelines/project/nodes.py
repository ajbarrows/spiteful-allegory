import os
import Bio

from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset

def establish_download_directory(fpath="data/01_raw/neuro-text/"):

    out_dir = os.path.abspath(fpath)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir

def download_and_convert_neurodata(out_dir, dataset:str, email="example@example.edu"):
    '''Save full dataset but return abstracts.'''

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

    n_dset = convert_neurosynth_to_dataset(
        coordinates_file=ndb["coordinates"],
        metadata_file=ndb["metadata"],
        annotations_files=ndb["features"],
        )
    n_dset = download_abstracts(n_dset, email)

    # save full dataset
    fname = f"{dataset}_dataset_with_abstracts.pkl.gz"
    n_dset.save(os.path.join(out_dir, fname))

    return n_dset.texts