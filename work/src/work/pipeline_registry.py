"""Project pipelines."""
from __future__ import annotations

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.project.pipeline import *

def register_pipelines() -> dict[str, Pipeline]:
    
    pipelines = {
        'download_data': download_data,
        'process_data': merge_neurodata
    }
    return pipelines
