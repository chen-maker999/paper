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
转换；组对象上的 `Members`、计算机对象上的管理员/RDP 列表会自动反向。未标注
的入口和目标不会由名称猜测。未知关系不会静默丢弃：可识别的未知类型会保留并
使用通用成本，统计信息位于 `instance.meta['import_stats']`。

## 抽象图生成

```bash
python -m experiments.ad_generator \
  --users 1000 --groups 80 --computers 300 --targets 5 \
  --seeds 0-4 --out data/ad_smoke
```

Python 接口为 `ADGeneratorConfig` 和 `generate_ad_graph`。生成器使用固定种子，
按每个源节点采样固定期望度数，输出的节点和关系是抽象实验数据，目录名包含
用户/组/计算机规模，能直接交给 `experiments.run`。

## 转换为实验 CSV

对 ADSynth 输出，入口和目标应由实验设计明确指定。准备两个文本文件，每行一个
节点 ID，可在第二列写权重或价值，然后运行：

```bash
python -m experiments.convert_ad \
  --input generated_datasets/vul_1k.json \
  --entries entries.txt --targets targets.txt \
  --out data/vul_1k_s0
```

转换器会写出 `edges.csv`、`nodes.csv` 和导入统计，并保留未知关系类型的统计信息。
