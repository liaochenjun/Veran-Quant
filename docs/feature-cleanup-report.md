# Feature Cleanup Report（8 个全 NaN 特征）

诊断对象：`behavior_dataset_full.json`（370 样本）的 train 部分（259 样本，全局时序切分 70%）。

## 定位结果

8 个全 NaN 特征 = **4 个 chan 周期 × 2 个字段**：

| feature_name | reason | affected_samples | affected_timeframes |
|--------------|--------|------------------|---------------------|
| chan__1m__divergence_strength | chan.py 开源 API 不单独暴露背驰强度，设计上恒为 None → NaN（masked） | 259/259（全部） | 1m |
| chan__5m__divergence_strength | 同上 | 259/259 | 5m |
| chan__15m__divergence_strength | 同上 | 259/259 | 15m |
| chan__1h__divergence_strength | 同上 | 259/259 | 1h |
| chan__1m__fractal_price | 开仓时刻最后一根合并 K 线几乎从未形成"已确认分型"（分型需下一根 K 线确认），fractal_present=False → price=None | 259/259 | 1m |
| chan__5m__fractal_price | 同上 | 259/259 | 5m |
| chan__15m__fractal_price | 同上 | 259/259 | 15m |
| chan__1h__fractal_price | 同上 | 259/259 | 1h |

affected_symbols：全部 27 个在 train 中有样本的 symbol（无一例外——这是结构性缺失，不是个别 symbol 的数据问题）。

## 结论

1. **不是 feature builder bug**：`divergence_*` 是 v1 明确 masked 的字段（chan.py 无此输出，见 docs/chan-integration.md §7/§11）；`fractal_price` 的缺失是"分型需要后续确认"这一缠论特性的必然结果——开仓时刻恰好处于已确认分型上的概率极低。`fractal_present/fractal_top/fractal_bottom`（one-hot）仍正常提供分型信息。
2. **处理方式**：在模型 `fit()` 时按 **train 集的观测**将全 NaN 特征从 feature matrix 中剔除（`_SklearnBehaviorModel._prepare_features`，train-only 决策，每个 walk-forward fold 独立重新判定）。**原始 Dataset 未做任何修改**——特征照常序列化、照常可追溯，只是模型不消费无信息列。
3. 影响：特征维度 200 → 192（全局 train 上）。

## 测试锁定

`tests/test_p15_discipline.py` 覆盖：全 NaN 特征被正确剔除、原始 Dataset 不被修改、每 fold 独立判定。
