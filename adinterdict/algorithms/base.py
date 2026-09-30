"""算法的公共返回类型。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    removed: set  # 原始实例上的删边编号集合
    cost: int
    info: dict = field(default_factory=dict)
