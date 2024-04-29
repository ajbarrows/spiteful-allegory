"""
This is a boilerplate pipeline 'project'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *
from .plotting import *

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([])

ingest_neurodata = pipeline(
    [
        node(
            establish_download_directory,
            inputs=None,
            outputs='out_dir'
        ),
        node(
            download_neurodata,
            inputs=['out_dir', 'dataset'],
            outputs='ndb'
        ),
        node(
            convert_dset,
            inputs='ndb',
            outputs='n_dset'
        ),
        node(
            download_neuro_abstracts,
            inputs=['n_dset', 'email'],
            outputs='n_dset_abstracts'
        ),
          node(
            save_neurodata,
            inputs=['n_dset_abstracts', 'dataset', 'out_dir'],
            outputs='n_dset_text'
        )
    ]
)


download_neurosynth = pipeline(
    pipe=ingest_neurodata,
    inputs={'dataset': 'params:neurosynth', 'email': 'params:email'},
    outputs={'n_dset_text': 'neurosynth_text'},
    namespace='neurosynth'
)

download_neuroquery = pipeline(
    pipe=ingest_neurodata,
    inputs={'dataset': 'params:neuroquery', 'email': 'params:email'},
    outputs={'n_dset_text': 'neuroquery_text'},
    namespace='neuroquery'
)

download_data = download_neurosynth + download_neuroquery


merge_neurodata = pipeline(
    [
        node(
            drop_duplicated_neuroquery,
            inputs=['neurosynth_text', 'neuroquery_text'],
            outputs=['ns', 'nq']
        ),
        # node(
        #     get_pubmed_citations,
        #     inputs='nq',
        #     outputs='nq_cite'
        # ),
        node(
            extract_dates_from_citations,
            inputs='nq_cite',
            outputs='nq_prepped'
        ),
        node(
            combine_texts,
            inputs=['ns', 'nq_prepped'],
            outputs='combined_text'
        )
    ]
)


process_text = pipeline(
    [   
        node(
            select_random_abstracts,
            inputs='neurosynth_text',
            outputs='random_abstracts'
        ),
        node(
            load_methods_corpus,
            inputs='methods_corpus',
            outputs='corpus'
        ),
        node(
            load_skip_words,
            inputs='params:skip_words',
            outputs='skip_words'
        ),
        node(
            text_to_dict,
            inputs='neurosynth_text',
            outputs='text'
        ),
        node(
            run_detect_parallel,
            inputs=['text', 'corpus', 'skip_words'],
            outputs='text_detect_list'
        ),
        node(
            gather_detected_output,
            inputs=['neurosynth_text', 'text_detect_list'],
            outputs='detected_text'
        )


        # node(
        #     rankcount_from_abstracts,
        #     inputs='neurosynth_text',
        #     outputs=['parsed', 'rankcount']
        # ),
        # node(
        #     add_nltk_pos,
        #     inputs=[
        #         'rankcount',
        #         'params:pos_to_keep',
        #         'params:simple_verbs'
        #     ],
        #     outputs='rankcount_pos'
        # )
    ]
)


generate_plots = pipeline(
    [
        node(
            plot_pubcounts,
            inputs='neurosynth_text',
            outputs=None
        ),
        node(
            make_rankcount_plot,
            inputs='rankcount_pos',
            outputs=None
        )
    ]

)