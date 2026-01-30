"""
Field utility functions for the data processing pipeline.

Functions that modify non-prompt fields in the row.
"""


def confirm_correct_answer(row: dict) -> dict | None:
    """
    If correct_answer is None and other_answers has exactly one element,
    set that as the correct_answer.
    """
    if row.get("correct_answer") is None and len(row.get("other_answers", [])) == 1:
        row = row.copy()
        row["correct_answer"] = row["other_answers"][0]
    
    return row
