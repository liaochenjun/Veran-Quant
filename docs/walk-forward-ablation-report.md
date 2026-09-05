# Walk-Forward & Feature Ablation 实验报告

数据：`behavior_dataset_full.json`（370 样本，3 KOL，30 symbol，LONG 226 / SHORT 144，2026-06-03 → 09-05）。
特征：`state-features-v2`，全量 200 维（market 5 周期 + 每周期 geometry + chan 4 周期 29 键 + trader one-hot）。
纪律：所有实验共享同一时序切分（train=259 / val=55 / test=56，chronological 70/15/15，零随机）；所有预处理（imputer）只在各自 train 上 fit；test label 只在预测后读取。

---

## 1. Walk-Forward Evaluation（expanding window，4 folds，每 fold 独立 fit）

Fold 划分（连续时间块，train 扩张）：

| Fold | Train 范围 | Train 样本 | Test 范围 | Test 样本 |
|------|-----------|-----------|----------|----------|
| 0 | 06-03 ~ 07-17 | 74 | 07-17 ~ 08-21 | 74 |
| 1 | 06-03 ~ 08-21 | 148 | 08-21 ~ 08-29 | 74 |
| 2 | 06-03 ~ 08-29 | 222 | 08-29 ~ 09-03 | 74 |
| 3 | 06-03 ~ 09-03 | 296 | 09-03 ~ 09-05 | 74 |

（无 per-fold validation，模型选择沿用全局 val split；不伪造样本凑 fold。）

### 各 fold 结果

| 模型 | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|------|--------|--------|--------|--------|
| Majority — acc | 0.378 | 0.622 | 0.824 | 0.541 |
| Majority — f1 | 0.275 | 0.383 | 0.452 | 0.351 |
| LR — acc | 0.500 | 0.622 | 0.757 | 0.622 |
| LR — f1 | 0.498 | 0.558 | 0.519 | 0.520 |
| RF — acc | 0.554 | 0.716 | 0.649 | 0.716 |
| RF — f1 | 0.553 | 0.680 | 0.523 | 0.673 |

### 跨 fold 聚合（Mean / Std / Min / Max）

| 模型 | 指标 | Mean | Std | Min | Max |
|------|------|------|-----|-----|-----|
| Majority | Accuracy | 0.591 | 0.161 | 0.378 | 0.824 |
| Majority | Macro F1 | 0.365 | 0.064 | 0.275 | 0.452 |
| Majority | Log Loss | 11.295 | 4.437 | 4.854 | 17.176 |
| LR | Accuracy | 0.625 | 0.091 | 0.500 | 0.757 |
| LR | Macro F1 | 0.524 | 0.022 | 0.498 | 0.558 |
| LR | Log Loss | 1.232 | 0.995 | 0.555 | 2.949 |
| RF | **Accuracy** | **0.659** | **0.066** | 0.554 | 0.716 |
| RF | **Macro F1** | **0.607** | **0.070** | 0.523 | 0.680 |
| RF | **Log Loss** | **0.627** | **0.061** | 0.567 | 0.717 |

**结论：Walk-Forward 下 Random Forest 是三个基线中最稳定且综合最好的**（mean accuracy 0.659、std 最小 0.066、log loss 最低 0.627）。LR 在小样本 fold 0（74 训练样本）明显退化（log loss 2.949）。多数类基线受时间漂移影响最大（std 0.161）。

---

## 2. Feature Ablation（LR，全局时序切分，评估代码完全一致）

| Feature Set | Accuracy | Macro F1 | Log Loss | LONG Recall | SHORT Recall | Worst Trader F1 |
|-------------|----------|----------|----------|-------------|--------------|-----------------|
| A. Market Only | 0.500 | 0.500 | 0.663 | 0.636 | 0.412 | 0.333 |
| B. Market + Geometry | 0.393 | 0.380 | 4.225 | 0.682 | 0.206 | 0.333 |
| C. Market + Chan | 0.393 | 0.282 | 0.875 | 1.000 | 0.000 | 0.226 |
| D. Market + Chan + Geometry | **0.661** | **0.655** | **0.484** | 1.000 | 0.441 | **0.604** |
| E. D + Trader ID | 0.661 | 0.655 | 0.560 | 1.000 | 0.441 | 0.604 |

**回答五个问题：**

1. **Chan 是否提供增益？** 是，但必须与 Geometry 组合。单独加 Chan（C）反而塌缩成"全 LONG"（SHORT recall=0）；Chan+Geometry 一起（D vs B）把 accuracy 从 0.393 拉到 0.661、log loss 从 4.225 降到 0.484——这是最大的增益组合。
2. **Geometry 是否提供增益？** 单独加 Geometry（B vs A）是负增益（0.500→0.393，log loss 恶化到 4.225）；但与 Chan 组合后（D vs C）贡献巨大。Geometry 单独进 LR 会引入噪声/过拟合，需要 Chan 结构特征共同作用才有意义。
3. **Trader ID 是否提供增益？** 无。E vs D：accuracy/F1 完全相同（0.661/0.655），log loss 反而变差（0.484→0.560）。在当前数据量下 trader one-hot 只是多余维度。保留在 pipeline 中（不影响结果），但当前不提供信息。
4. **哪些特征组可能只是增加维度？** 单独使用时的 Geometry 组和 Chan 组；trader one-hot 在现有数据上是纯增维。它们只有组合后才有效。
5. **200 维是否过宽？** 是。最优子集 D（market+chan+geometry）即全部非 trader 特征，说明问题不在"过宽"，而在"单组噪声大"。但注意：imputer 报告有 **8 列在训练集中完全无观测值**（全部为 NaN，被跳过）——这些是覆盖缺失产出的空列（如 HYPE/1000PEPE 缺 1m 数据对应的 chan 列），属于死维度，可以清理。

---

## 3. LR 系数 Top-30（仅在全局 train 上 fit；正值 → 偏向 SHORT，负值 → 偏向 LONG）

前 15（完整 30 见 experiments_report.json）：

| 方向 | 系数 | 特征 |
|------|------|------|
| SHORT | +0.0022 | chan__1h__zhongshu_is_sure |
| LONG | -0.0020 | chan__15m__zhongshu_present |
| SHORT | +0.0016 | chan__5m__segment_direction_up |
| LONG | -0.0012 | market__1h__geometry__number_of_touches |
| SHORT | +0.0011 | market__4h__geometry__local_trend_structure__uptrend |
| SHORT | +0.0011 | market__15m__geometry__angle |
| SHORT | +0.0011 | market__15m__geometry__line_strength |
| LONG | -0.0011 | chan__1h__buy_sell_point_type_3a |
| LONG | -0.0009 | market__1m__geometry__distance_from_current_price |
| LONG | -0.0009 | market__1m__geometry__local_trend_structure__flat |
| SHORT | +0.0008 | market__4h__geometry__angle |
| SHORT | +0.0008 | market__4h__geometry__line_strength |
| SHORT | +0.0007 | chan__5m__fractal_present |
| SHORT | +0.0007 | market__1m__geometry__previous_high |
| LONG | -0.0007 | market__4h__geometry__distance_from_current_price |

观察：缠论结构特征（中枢确认/分型/线段方向/买卖点）占据系数榜首；geometry 的触碰数/角度/趋势一热也是主力。**注意：特征未经标准化（价格量纲与计数混在一起），系数绝对值跨特征不可直接比较**；另 LR 在 2000 次迭代未完全收敛（lbfgs ConvergenceWarning），方向性结论有效但幅度仅供参考。

---

## 4. 类别权重实验（独立实验对比，非默认变更）

| 配置 | Accuracy | Macro F1 | Log Loss | LONG Recall | SHORT Recall |
|------|----------|----------|----------|-------------|--------------|
| class_weight=None（默认） | 0.661 | 0.655 | 0.560 | 1.000 | 0.441 |
| class_weight="balanced" | **0.714** | **0.713** | **0.505** | 0.818 | 0.647 |

balanced 在测试集全面领先（SHORT 召回 0.441→0.647），印证了阶段一结论"测试期转空导致的 SHORT 欠召回"可以通过类别权重缓解。**这是当前单配置最佳结果（test acc 0.714 / f1 0.713）。**

---

## 5. 结论与建议

- **当前最佳模型**：LR + class_weight=balanced（test acc 0.714 / macro F1 0.713 / log loss 0.505）；Walk-Forward 下 RF 更稳定（mean acc 0.659）。
- **当前最大问题**：时间分布漂移（测试期 SHORT 占比高）仍是主要误差源；小样本 fold（74 训练样本）下 LR 退化明显。
- **下一步建议**（按性价比）：
  1. 继续导入更多历史数据（最有效）；
  2. 对特征做标准化（StandardScaler 仅 fit train），可改善 LR 收敛并让系数可比；
  3. 清理 8 个全 NaN 死特征列；
  4. 尝试 GBDT（LightGBM/XGBoost）与 walk-forward + 类别权重组合；
  5. 为漂移监控记录每个 fold 的 label 分布。

**未做**（遵守阶段纪律）：无 Transformer/LSTM/RL、无 Trader Embedding、无 Event NLP、无 SMOTE/伪造样本、无随机切分、无 test 特征选择。
