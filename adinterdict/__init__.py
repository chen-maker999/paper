"""adinterdict：预算约束下的 AD 攻击路径阻断（CMTI 问题）实验框架。"""
from .indexed import IndexedInstance
from .model import INF, Instance, build_graph

__all__ = ["INF", "Instance", "IndexedInstance", "build_graph"]
