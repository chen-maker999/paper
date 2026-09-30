# AD 图导入与合成

这些工具用于离线实验。它们处理的是已经导出的节点和关系，不连接域控制器、不
执行目录操作，也不包含凭据处理。

## 规范化输入

```json
{
  "nodes": [
    {"id": "user-1", "type": "User", "entry_weight": 1},
    {"id": "group-1", "type": "Group"},
    {"id": "target-1", "type": "Group", "target_value": 10}
  ],
  "edges": [
    {"source": "user-1", "target": "group-1", "type": "MemberOf"},
    {"source": "group-1", "target": "target-1", "type": "GenericAll"}
  ]
}
```

```python
from adinterdict.adio import load_ad_export
instance = load_ad_export("graph.json")
```

入口和目标也可以在调用时指定：

```python
instance = load_ad_export(
    "graph.json",
    entry_weight={"user-1": 1.0},
    target_value={"target-1": 10.0},
)
```

目录输入会合并其中的 `*.json` 文件。节点记录支持 `id`、`type`、`Properties`、
`ObjectIdentifier` 等常见字段；关系记录支持 `source`/`target` 和
`SourceNodeId`/`TargetNodeId`。节点属性中的列表关系（例如 `MemberOf`）也会被
转换。未标注的入口和目标不会由名称猜测。

## 抽象图生成

```bash
python -m experiments.ad_generator \
  --users 1000 --groups 80 --computers 300 --targets 5 \
  --seeds 0-4 --out data/ad_smoke
```

Python 接口为 `ADGeneratorConfig` 和 `generate_ad_graph`。生成器使用固定种子，
输出的节点和关系是抽象实验数据，能直接交给 `experiments.run`。
