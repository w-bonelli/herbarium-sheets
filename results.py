from typing import TypedDict


class AnalysisResult(TypedDict, total=False):
    name: str
    area: float
    width: float
    height: float
    length: int
    leaves: int
    branch_points: int
    end_points: int