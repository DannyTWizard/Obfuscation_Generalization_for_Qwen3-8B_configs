"""
Data processing functions for the pipeline.

Each function takes a row (dict) and keyword arguments from the config.
Returns a transformed dict, or None to filter out the row.

Row schema (after prompt_list -> prompt conversion):
    - prompt: str
    - high_reward_answer: str
    - other_answers: list[str]
    - correct_answer: str | None
    - source_dataset: str
"""

from data_processing.functions.filters import *
from data_processing.functions.text_edits import *
from data_processing.functions.field_utils import *
from data_processing.functions.xml_formatting import *
