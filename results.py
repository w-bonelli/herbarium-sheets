from typing import TypedDict


class AnalysisResult(TypedDict, total=False):
    name: str
    area: float
    max_width: float
    max_height: float
    leaves: int