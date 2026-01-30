"""
Filtering functions for the data processing pipeline.

Each function takes a row (dict) and returns the row to keep it,
or None to filter it out.
"""


def revealing_score_adversarial(row: dict) -> dict | None:
    """
    Filter to adversarial examples where high_reward_answer != correct_answer.
    
    Only valid for revealing_score dataset.
    """
    if row["source_dataset"] != "revealing_score":
        raise ValueError(
            f"revealing_score_adversarial can only be applied to revealing_score dataset, "
            f"got: {row['source_dataset']}"
        )
    
    if row["high_reward_answer"] == row["correct_answer"]:
        return None
    
    return row