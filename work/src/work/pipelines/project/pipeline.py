"""
This is a boilerplate pipeline 'project'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([])

download_data = pipeline(
    [
        node(
            establish_download_directory,
            inputs=None,
            outputs='out_dir'
        ),
        # node(
        #     download_and_convert_neurodata,
        #     inputs=['out_dir', 'params:neurosynth', 'params:email'],
        #     outputs='neurosynth_text'
        # ),
        node(
            download_and_convert_neurodata,
            inputs=['out_dir', 'params:neuroquery', 'params:email'],
            outputs='neuroquery_text'
        )
    ]
)