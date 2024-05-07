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


    ]
)

model_pipeline = pipeline(
    [
        node(
            add_cleaned_text_column,
            inputs='detected_text',
            outputs=['ns_prepped', 'docs_prepped']
        ),
        node(
            fit_BERTopic_model,
            inputs='docs_prepped',
            outputs='BERTopic_model'
        ),
        node(
            extract_BERTopic_topics,
            inputs=['BERTopic_model'],
            outputs='BERTopic_topics'
        ),
        node(
            fit_lda_model,
            inputs='docs_prepped',
            outputs=['lda_model', 'vectorizer']
        ),
        node(
            extract_LDA_topics,
            inputs=['lda_model', 'vectorizer'],
            outputs='lda_topics'
        ),
        node(
            join_output,
            inputs=['BERTopic_topics', 'lda_topics'],
            outputs='overall_topics'
        ),
        node(
            fit_models_by_group,
            inputs=['detected_text', 'params:year_map'],
            outputs=['topics_by_year', 'lda_group_models', 'bert_group_models']
        )
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
            rankcount_from_abstracts,
            inputs='detected_text',
            outputs=['parsed', 'rankcount']
        ),
        node(
            add_nltk_pos,
            inputs=[
                'rankcount',
                'params:pos_to_keep',
                'params:simple_verbs'
            ],
            outputs='rankcount_pos'
        ),
        node(
            make_rankcount_plot,
            inputs="rankcount_pos",
            outputs=None
        )
    ]

)