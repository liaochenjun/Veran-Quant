# P2-1.5：Behavior Prediction Evaluation Redesign 报告（真实结果）

数据：data_version `0cb9dbfaa775b10b`（时区修正后正式 Dataset，370 样本，3 KOL）。所有分析仅使用 T 之前可见信息；Leakage Detector 实验前 PASS（370/370）。方法全部预先确定、不使用 outcome、不按 Test 结果调任何东西。

## 1. 官方 Walk-Forward（保留为时间泛化基准）

4 折扩张窗口，RF（n_estimators=300，每折独立重训）：**mean Macro F1 = 0.619，mean LogLoss = 0.575**（逐折见 summary.json）。

## 2. Multiple Test Windows（自动生成 4 个等宽窗口；窗口 0 因 train 样本为 0 被 SKIPPED）

| 窗口 | 范围 | Train | Test | Acc | Macro F1 | LogLoss | LONG Recall | SHORT Recall | Worst Trader F1 |
|------|------|-------|------|-----|----------|---------|-------------|--------------|-----------------|
| 1 | 06-26 ~ 07-19 | 41 | 35 | **0.686** | **0.685** | 0.606 | 0.71 | 0.64 | 0.44 |
| 2 | 07-19 ~ 08-12 | 76 | 30 | 0.533 | 0.525 | 0.663 | 0.64 | 0.25 | 0.22 |
| 3 | 08-12 ~ 09-04 | 106 | 264 | 0.424 | 0.366 | 0.703 | 0.47 | 0.26 | 0.20 |

**时间泛化退化明显**：越靠后的窗口性能越差（f1 0.685 → 0.525 → 0.366），窗口 3 接近甚至低于多数类水平。这与"时间越远、行为分布越不同"一致，但**不能直接归因为 drift**（见 §6）。

## 3. Market Regime 结果（趋势：4h→1h→15m geometry one-hot；波动：1h 已实现波动率，早半段中位数切分）

| Regime | n | Actual LONG Ratio | 模型 Mean P(LONG) | Gap |
|--------|---|-------------------|-------------------|-----|
| trend_up | 209 | 0.512 | 0.525 | +0.013 ✓ |
| trend_down | 158 | **0.741** | 0.573 | **-0.168** |
| sideways | 3 | 0.667 | 0.067 | （样本太少） |
| high_volatility | 153 | **0.699** | 0.529 | **-0.171** |
| low_volatility | 214 | 0.547 | 0.556 | +0.009 ✓ |

**关键发现**：这些 KOL 在**下跌趋势与高波动**状态下反而更倾向 LONG（实际 0.74 / 0.70）——逆向/抄底风格；模型在这两类 regime 上系统性低估 LONG（gap -0.17）。trend_up 和低波动下模型校准良好。这就是"模型没有充分理解 Market State → Action 关系"的直接证据：**它学到的是平均行为，而不是 regime 条件下的行为**。

## 4. Market State → Action（同上表：Actual vs Predicted distribution 对比，已包含）

## 5. Same-State / Different-Time（方法：早半段标准化特征空间，k=5 最近邻状态匹配，比较邻域 LONG 比例 vs 后期实际 LONG 比例）

| Trader | Early LONG | Late LONG | 状态匹配邻域 LONG | 状态匹配后期 LONG | 状态匹配差 |
|--------|-----------|-----------|-------------------|-------------------|-----------|
| 全部 | 0.551 | 0.670 | 0.536 | 0.700 | **+0.164** |
| 熬鹰资本 | 0.444 | 0.695 | 0.476 | 0.690 | **+0.213** |
| 刘元 | 0.500 | 0.714 | 0.419 | 0.533 | +0.114 |
| 西九夜 | 0.711 | 0.590 | 0.546 | 0.431 | **-0.115** |

## 6. 是否存在 Behavior Drift？

**存在方向性证据（weak / directional），但不能做强结论。** 状态匹配后（控制了市场形态）：
- 熬鹰与刘元在相似状态下后期系统性**更偏 LONG**（+0.21 / +0.11）；
- 西九夜在相似状态下后期系统性**更偏 SHORT**（-0.12）——与你观察到的"9 月转空"一致，且是在相似 K 线状态下成立的，因此这次可以称为方向性 drift 证据；
- 样本量：后期共 ~185 样本、每 trader 20~80，200 维 kNN 匹配的方差大——证据强度为 **weak / directional**，不足以定量断言。

## 7-10. 回答

1. 多时间窗口：见 §2（后窗退化）。
2. Walk-Forward：mean F1 0.619。
3. Regime：见 §3（trend_down/高波动下模型低估 LONG）。
4. State→Action：见 §4。
5. Same-State/Different-Time：见 §5。
6. Drift：方向性证据（熬鹰/刘元偏多、西九夜偏空，状态匹配后仍成立）。
7. 有统计支持的结论：**时间泛化退化**（窗口 1→3 单调下降，4 折 WF 均验证）；**regime 校准缺口**（trend_down/high_vol 下 gap ≈ -0.17）。
8. 仅方向性证据：per-trader drift、regime 内 per-trader 差异。
9. 数据限制：370 样本 / 93 天；sideways 仅 3 样本；后期窗口 traDDer 分布不均；kNN 匹配在高维空间方差大。
10. 下一阶段建议：① **regime 特征显式入模**（trend/vol 正是模型缺的上下文——与架构 §9 Event Context 方向一致，这是目前最强的可操作发现）；② 继续收集数据（drift 分析需要更多样本才能升级证据强度）；③ per-trader 模型（熬鹰 vs 西九夜 drift 方向相反，共享模型互相拖累）。

## 输出文件
`data/experiments/p2_1_5_evaluation/{window_metrics,window_trader_metrics,window_action_metrics,regime_metrics,state_shift_analysis}.csv + summary.json`

pytest：见最终运行结果。
