from typing import Literal, Tuple, TypeAlias, get_args

Action: TypeAlias = Literal[
    "new_angle",
    "deepen",
    "criticize",
    "revise",
    "final_report",
    "final_review",
    "answer",
]
ACTIONS: Tuple[Action, ...] = get_args(Action)
