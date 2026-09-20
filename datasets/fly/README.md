# 果蝇默认数据集

当前 MaleCNS 实例的预训练产物。发布给不自己拟合的使用。`artifacts/aCC.npz`、`type_0.npz`、`hh_demo.npz` 仍是种子，不在这里改。

| 目录 | 内容 |
|---|---|
| `fits/` | 各类型的 L0、L1、L2、L1+L2。表是 float64。 |
| `precision/` | 同一张表的 float32 网格。边权在 `sparse/shiu_labelled_circuit.npz` 里同时有 float32 和 float64。 |
| `sparse/` | 附属屈肌运动神经元及其最强上游组成的回路。全量 Feather 仍在 `data/malecns/`。 |
| `reports/` | NFR-4 / NFR-5、chirp、α、闭环。L3（FNO）只记录为 v0.2，没有伪造产物。 |
| `prefer/` | 可选。同一上限下效率 / 中性 / 拟真三份表。不在默认 `pretrained` 档里。 |

重新生成：

```bash
uv run --no-sync python scripts/export_fly_default_dataset.py
uv run --no-sync python scripts/export_prefer_datasets.py
```

校验默认档，以及可选的效率 / 中性 / 拟真档：

```bash
uv run --no-sync nrde fetch fly --tier pretrained
uv run --no-sync nrde fetch fly --tier prefer
```

README 上的精度 / 墙钟图从这些表重算，审计见 [证据链](../../docs/reports/evidence.md)。
