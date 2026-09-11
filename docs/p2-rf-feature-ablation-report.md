# P2-0：RF Feature Ablation 报告（真实结果）

> ⚠️ LEGACY：本报告基于时区修复前（KOL 时间戳被误当 UTC）的数据集，模型数字已作废。
> 保留仅为方法参考；正式结论见 P2-1 / P2-1.5 报告。

前置：P1.5 已冻结（commit `7772821`，tag `p1.5-strong-baseline`）。本阶段复用 P1.5 框架：同一数据集（data_version `31699502605ae8b4`，feature_version `state-features-v2`）、同一时序切分（259/55/56）、同一 4 折 expanding walk-forward、同一 RF 参数（n_estimators=300、默认权重、random_state=0）。Leakage Detector 实验前 PASS（5/5）。仅特征子集变化。

## 1. Global 70/15/15（每个特征集的 Test 结果）

| Feature Set | n | Val Acc | Val F1 | Test Acc | Test F1 | Test LogLoss | LONG Recall | SHORT Recall | Worst Trader F1 |
|-------------|---|---------|--------|----------|---------|--------------|-------------|--------------|-----------------|
| A Market Only | 15 | 0.564 | 0.564 | 0.554 | 0.553 | 0.820 | 0.955 | 0.206 | 0.250 |
| B Market+Geometry | 80 | 0.600 | 0.599 | 0.554 | 0.553 | 0.678 | 1.000 | 0.176 | 0.375 |
| C Market+Chan | 131 | 0.618 | 0.615 | 0.536 | 0.505 | 0.822 | 1.000 | 0.118 | 0.250 |
| D Market+Chan+Geometry | 196 | 0.600 | 0.594 | **0.625** | **0.615** | 0.755 | 1.000 | 0.235 | 0.333 |
| E Full no Trader（=D） | 196 | 0.600 | 0.594 | 0.625 | 0.615 | 0.755 | 1.000 | 0.235 | 0.333 |
| F Full（+Trader） | 200 | 0.600 | 0.596 | 0.607 | 0.594 | 0.775 | 1.000 | 0.353 | 0.541 |

## 2. Walk-Forward（4 折，逐折 Macro F1 / LogLoss）

| Feature Set | F0 f1 | F1 f1 | F2 f1 | F3 f1 | **Mean f1** | F0 ll | F1 ll | F2 ll | F3 ll | **Mean ll** |
|-------------|-------|-------|-------|-------|---------|-------|-------|-------|-------|---------|
| A Market Only | 0.581 | 0.546 | 0.497 | 0.679 | 0.576 | 0.789 | 0.709 | 0.829 | 0.697 | 0.756 |
| B +Geometry | 0.553 | 0.539 | 0.495 | 0.644 | 0.558 | 0.733 | 0.634 | 0.682 | 0.527 | 0.644 |
| C +Chan | 0.461 | 0.612 | 0.474 | 0.495 | 0.510 | 0.738 | 0.591 | 0.650 | 0.762 | 0.685 |
| D +Chan+Geometry | 0.513 | 0.641 | 0.490 | 0.580 | 0.556 | 0.723 | 0.569 | 0.578 | 0.658 | 0.632 |
| E Full no Trader | 同 D | | | | 0.556 | | | | | 0.632 |
| **F Full** | 0.554 | 0.692 | 0.552 | 0.673 | **0.617** | 0.715 | 0.565 | 0.584 | 0.654 | **0.630** |

## 3. Feature Delta（Δ vs Market Only，Walk-Forward 逐折）

| 对比 | Mean ΔAcc | Mean ΔF1 | Mean ΔLogLoss | ΔF1 逐折 | ΔLogLoss 逐折 |
|------|-----------|----------|---------------|----------|---------------|
| B − A（Geometry） | -0.007 | -0.018 | -0.112 | -0.028/-0.007/-0.001/-0.035（4负） | 4 折全负 |
| C − A（Chan） | 0.000 | -0.065 | -0.071 | -0.120/+0.066/-0.023/-0.184（3负） | 3 负 1 正 |
| D − A（Chan+Geo） | +0.017 | -0.020 | -0.124 | -0.068/+0.095/-0.007/-0.099（3负） | 4 折全负 |
| F − A（Full+Trader） | +0.071 | +0.042 | -0.127 | -0.027/+0.145/+0.055/-0.007（2正2负） | **4 折全负** |

## 4. 特征质量审计（train-only，只报告不删除）

- 全 NaN：8（`divergence_strength`×4、`fractal_price`×4，见 P1.5 清理报告）
- **常数特征：17**——`chan__<tf>__fractal_present/top/bottom`、`chan__<tf>__divergence_present` 共 16 个（train 中这些结构从未在开仓时刻出现，恒为 0）+ `trader__other`（恒 0，train 中无未知 trader）
- 近常数：12（`segment_is_sure` 恒 1、`number_of_breaks` 恒 ~0、`local_trend_structure__flat` 恒 ~0）
- **高相关对：752 个（|r|>0.95）**，其中 1.000 相关的结构性重复：`support_line == previous_low`、`resistance_line == previous_high`（几何提取器按定义相等）、`line_strength ≈ number_of_touches`（breaks≈0）、`uptrend` 与 `downtrend` one-hot 互为补集

## 5. RF Feature Importance（Full，train-only）

类别计数：chan 108 / geometry 65 / market 15 / trader 4。Top：angle、slope、zs_count、segment_count、distance_from_current_price、distance_to_zhongshu——树模型确实大量消费 chan/geometry 特征（importance 高不代表有增益，见下）。

## 6. 明确回答 12 个问题

1. **RF 上 Market Only 是否最优？** 否。WF mean f1 0.576，低于 Full 的 0.617；但其 log loss 最差（0.756），且 15 维最简单。
2. **Geometry 单独有没有增益？** 无正增益。ΔF1 mean -0.018（4 折全负）；但 ΔLogLoss 4 折全负（校准变好、分类变差）。
3. **Chan 单独有没有增益？** 无。ΔF1 mean -0.065（3/4 折为负），最差组合。
4. **Chan + Geometry 有没有交互增益？** 无明确交互。D（0.556）不优于 B（0.558），仅优于 C（0.510）——只说明 Chan 单独最差，组合后回到 Geometry 水平。
5. **Full 是否超过 Market Only？** WF mean f1 是（+0.042），且 **log loss 在全部 4 折稳定改善**（-0.127 mean）；但 ΔF1 逐折 2 正 2 负，**不能称为稳定增益**。
6. **Trader ID 是否有增益？** 有方向性但不一致：WF mean f1 0.556→0.617（+0.061），但 Global Test 0.615→0.594（-0.021）。证据不足，不能下"有增益"结论。
7. **Chan/Geometry 的增益是否跨 folds 稳定？** 不稳定。F1 增益逐折符号混杂；唯一稳定的是 **log loss 的改善**（Full vs Market Only 4/4 折为负）。
8. **哪些特征类别真正贡献泛化性能？** 结论：Chan+Geometry 主要贡献**概率校准**（log loss），对排序/准确率的贡献小且不稳定；Trader 在 WF 有帮助但在 Global Test 反向。真正的分类信号大部分已在 15 个基础 market 特征中。
9. **当前是否有理由继续优化 Chan/Geometry？** 没有强理由。当前证据只支持"保留但不优化"；更值得做的是特征去重（752 对高相关、17 常数、12 近常数）。
10. **当前 Champion 是否仍然是 RF？** 是。P2-0 没有引入新模型；RF（Full）仍是 WF 与综合性能最好的配置。
11. **当前是否有理由进入 Trader Embedding？** 无。trader one-hot 增益方向不一致，样本 370，主要误差源是时间漂移。
12. **当前是否应该继续模型复杂化？** 否。保持 RF，优先：特征去重/精简、数据扩充、漂移监控。

## 7. 决策规则落点

- 情况 A（Market Only ≈ Full）：**部分成立**——f1 差距小且不稳，但 log loss 差距一致。
- 情况 B（Geometry 单独稳定增益）：**不成立**。
- 情况 C（Chan×Geometry 交互增益）：**不成立**。
- 情况 D（Full 明显优于所有简化集且稳定）：**部分成立**——mean f1 最高 + log loss 全折改善，但 f1 折间不稳。
- **最终判定：以 F（Full）为当前特征集（mean f1 最高、log loss 最低且稳定改善），但必须如实声明其 f1 增益并不折间稳定；Chan/Geometry 的贡献主要是校准而非分类。下一步优先做特征去重审计落地，而非继续加特征。**

## 8. 输出与测试

- JSON：`data/processed/p2_rf_feature_ablation.json`（含 experiments/folds/deltas/importance/audit/champion）
- pytest：见最终运行结果（136+）
- 本阶段未 commit（按纪律，等你确认）
