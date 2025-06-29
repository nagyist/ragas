"""Base class from which all discrete metrics should inherit."""

__all__ = ["discrete_metric", "DiscreteMetric"]

import typing as t
from dataclasses import dataclass, field

from pydantic import create_model

from . import Metric
from .decorator import create_metric_decorator


@dataclass
class DiscreteMetric(Metric):
    values: t.List[str] = field(default_factory=lambda: ["pass", "fail"])

    def __post_init__(self):
        super().__post_init__()
        values = tuple(self.values)
        self._response_model = create_model(
            "response_model", result=(t.Literal[values], ...), reason=(str, ...)
        )


discrete_metric = create_metric_decorator(DiscreteMetric)
