# 果蝇默认数据集

当前 MaleCNS 实例的预训练产物。发布给不自己拟合的使用。`artifacts/aCC.npz`、`type_0.npz`、`hh_demo.npz` 仍是种子，不在这里改。

这些表是**类型级**降阶曲线。脑区和位置没有写进 npz，增加身份轴占位之后不必重训。首页数字从这些表重算，见仓库 [README](../../README.md) 与 [证据链](../../docs/reports/evidence.md)。

| 目录 | 内容 |
|---|---|
| `fits/` | 各类型的 L0、L1、L2、L1+L2。表是 float64。 |
| `precision/` | 同一张表的 float32 网格。边权在 `sparse/shiu_labelled_circuit.npz` 里同时有 float32 和 float64。 |
| `sparse/` | 附属屈肌运动神经元及其最强上游组成的回路。全量 Feather 仍在 `data/malecns/`。 |
| `reports/` | NFR-4 / NFR-5、chirp、α、闭环。L3（FNO）只记录为 v0.2，没有伪造产物。 |
| `prefer/` | 可选。同一上限下效率 / 中性 / 拟真三份表。不在默认 `pretrained` 档里。 |

校验默认档，以及可选的效率 / 中性 / 拟真档：

```bash
uv run nrde fetch fly --tier pretrained
uv run nrde fetch fly --tier prefer
```

克隆后这两条只校验哈希。表不在工作区、且清单里写了下载地址时，才会下载。
