# Chan (缠论) Integration

本文档记录将 [Vespa314/chan.py](https://github.com/Vespa314/chan.py) 集成进本项目的设计、约束与已知差异。

## 1. 为什么选择 chan.py

- 成熟、社区活跃的开源缠论实现（笔 / 线段 / 中枢 / 买卖点 / 背驰判定）。
- **原生支持逐根 K 线增量投喂**（`CChan.trigger_load`），官方文档明确给出防未来信息的使用方式（quick_guide.md「如何防止未来信息」一节），与本项目 point-in-time 的核心要求天然契合。
- 元素带有 `is_sure`（是否已确定）标记，可以区分已确认结构与未完成结构，而不是把未完成结构当最终结果。
- MIT License，可自由用于研究/商业用途。

不选择自研缠论的原因：缠论实现细节繁多（K 线合并、分型、特征序列线段、中枢扩展等），自研成本高且容易在细节上与规范产生偏差。

## 2. 使用的版本 / commit

- 仓库：`https://github.com/Vespa314/chan.py`
- 集成方式：git submodule（`third_party/chan.py`），**固定 commit**：

  ```
  429d6ed3043e27c93a003ba2b10e70a05575e1f5
  ```

  初始化：`git submodule update --init -- third_party/chan.py`

- chan.py 未发布 PyPI 包（PyPI 上的 `chan` 包是无关项目 stuglaser/pychan，勿装），仓库自身也没有 setup.py/pyproject，因此采用 submodule 固定 commit 而非 pip 依赖。
- 运行依赖：`numpy`、`pandas`（已加入 requirements.txt；matplotlib/ipython 仅画图用，本项目不需要）。
- Python 版本：chan.py master 使用 `typing.Self`（3.11+）。本项目运行于 3.10，`src/chan/chan_adapter.py` 在导入 chan.py 前用 `typing_extensions` 回填 `typing.Self`（`_patch_typing_self`），全库仅此一处 3.11 依赖（已核对：`Self` 仅用于类型注解，无其他 3.11 语法）。升级到 3.11+ 后该 shim 自动失效。

## 3. License

chan.py 为 **MIT License**（Copyright (c) 2022 Memos）。合规要求：保留上游版权声明。本项目不复制 chan.py 源码（submodule 引用上游仓库，自带 LICENSE 文件），仅在 `docs/chan-integration.md`（本文）与代码注释中记录来源与版本。

## 4. Adapter 架构

```
src/chan/
├── chan_engine.py       # 既有：ChanEngine 抽象接口 + SimpleChanEngine 占位（未改动）
├── chan_adapter.py      # chan.py 隔离层：sys.path bootstrap + Self shim + ChanPyAdapter
├── chan_state.py        # ChanState：稳定、可序列化的状态快照（默认值 + mask）
├── causal_chan.py       # CausalChanEngine：point-in-time 安全的 ChanEngine 实现
└── feature_encoder.py   # ChanFeatureEncoder：ChanState -> 数值特征向量
```

- `ChanPyAdapter`：封装一个 `CChan` 实例（单周期），提供 `feed_bar()`（增量投喂已收盘 K 线）与 `snapshot()`（冻结为 `ChanState`）。外部 API 变化只影响本文件。
- `CausalChanEngine`：实现既有 `ChanEngine.get_state()` 接口，可直接替换 `SimpleChanEngine` 传给 `BehaviorDataset.from_trades`。
- `ChanState`：模型只消费该稳定结构，**绝不直接接触 chan.py 的内部对象**（CBi/CSeg/…）。

## 5. Causal / Point-in-Time 设计

- **数据源边界**：K 线全部来自 `DuckDBStorage.read_klines(..., end_before=as_of)`，SQL 层即强制 `close_time < as_of`（严格小于，未收盘 K 线不可见）。
- **增量计算**：使用 chan.py 官方推荐的 `trigger_load` + `trigger_step=True` 模式：
  - `CChan.__init__` 不加载任何历史数据（全部由我们逐根投喂）；
  - 每根 K 线投喂后，chan.py 只用「目前已投喂的数据」计算状态 —— 这是库层面的因果保证（quick_guide「如何防止未来信息」）。
- **冻结语义**：`snapshot()` 把当前状态复制为普通数据（`ChanState`），返回后不再与 chan.py 内部可变对象有任何引用关系。之后投喂新 K 线只会影响 chan.py 内部对象与**未来**的快照，不会改写已返回的快照。
- **增量缓存**：每个 (symbol, timeframe) 维护一个 adapter，按时间单调前进；只投喂 `close_time` 晚于上次投喂的新 K 线。
- **向后查询**：若查询的 `as_of` 早于已投喂的最后一根 K 线（例如回测随机访问），engine 用「该时刻可见的 K 线」重建 adapter —— 保证 T1 时刻的状态永远只由 ≤ T1 的可观测数据产生。
- 结论：T1 的状态不会因为后来出现 T2/T3/T4 的 K 线而改变。

## 6. 多周期设计

- chan.py `KL_TYPE` 枚举没有 4h 级别（`K_1M/K_3M/K_5M/K_10M/K_15M/K_30M/K_60M/K_DAY/...`）。v1 支持 **1m / 5m / 15m / 1h** 四个级别，**4h 返回 `supported=False` 的全 mask 状态**（不伪造、不把 4h 硬塞进错误级别）。
- 每个周期一个**独立** `CChan` 实例（`lv_list=[该级别]`）：
  - chan.py 的多级别联立（父子级别）要求每次 `trigger_load` 都包含最高级别数据，与「1m K 线先于 1h K 线出现」的增量节奏冲突；
  - 独立实例天然因果：每个周期的状态只由该周期 ≤ T 的已收盘 K 线决定。
- 每个周期的状态快照在 `as_of` 时刻被冻结，冻结的是当时可观察到的状态（含 `is_sure=False` 的未完成结构）。
- 未来若要恢复 chan.py 多级别联立（区间套），需要按最高级别节奏批量触发，届时再评估。

## 7. ChanState 数据结构

`src/chan/chan_state.py` 的 `ChanState`（dataclass，slots）：

| 字段 | 说明 |
|------|------|
| symbol / timeframe / as_of_timestamp | 标识 + 冻结时刻（ISO，aware UTC） |
| supported | 该 timeframe 是否被 chan.py 支持（4h = False） |
| last_bar_close_time | 快照可见的最后一根 K 线收盘时间 |
| fractal_present / fractal_type / fractal_price | 最后一根合并 K 线的分型（TOP/BOTTOM；未确认 = present False） |
| bi_count / bi_direction / bi_is_sure / bi_amplitude / bi_length | 最后一笔（方向、是否确定、幅度、覆盖合并K线数）+ 笔总数 |
| segment_count / segment_direction / segment_is_sure | 最后一线段 + 总数 |
| zs_count / zhongshu_present / zhongshu_high / zhongshu_low / zhongshu_is_sure / distance_to_zhongshu | 最后中枢 + 距离（>0 在上方，<0 在下方，0 在内部） |
| divergence_present / divergence_type / divergence_strength | 背驰：chan.py 开源 API 未单独暴露 → v1 恒 masked |
| buy_sell_point_present / buy_sell_point_types / buy_sell_point_is_buy / buy_sell_point_bi_is_sure / buy_sell_point_time | 最新买卖点（chan.py 没有 bsp 级 is_sure，用所属笔的 is_sure 代理并如实命名） |

**约定**：结构不存在 → 值字段 `None` + `*_present=False`；**绝不把缺失结构当 0**。`to_dict()` 键序稳定（字段声明序），`from_dict()` 可逆。

## 8. Feature Encoder

`src/chan/feature_encoder.py`：

- `encode(state)`：单个 ChanState → 定长有序数值 dict（`FEATURE_KEYS` 固定 29 个键）。mask/计数恒为数值；缺失结构的值字段为 **NaN**（下游模型自行处理，绝不静默填 0）；方向编码 `UP=1/DOWN=0`；买卖点类型按 1/1p/2/2s/3a/3b 做 one-hot。
- `encode_multi(states)`：多周期字典 → `"<timeframe>__<feature>"` 前缀的平铺特征。
- 未来 Behavior Cloning 模型的特征 = Geometry（支撑/阻力/斜率/角度/距离/触碰/突破/趋势，见 `src/market/geometry.py`，保持不变）+ 上述 Chan 特征 + 多周期组合。

## 9. Future Leakage 防护（逐条核对）

| 泄漏路径 | 防护 |
|----------|------|
| 使用 T 之后的 K 线 | storage SQL 层 `close_time < as_of` 严格过滤；engine 只投喂过滤后的 K 线 |
| 使用未收盘 K 线的最终 OHLC | 未收盘 K 线 close_time ≥ as_of，根本不会被读到（测试：`test_bar_closing_at_as_of_is_excluded`） |
| 使用未来 high/low/close | 同上，K 线整体不可见 |
| 未来确认后的缠论结构回填过去 | 快照冻结 + 向后查询重建（测试：`test_future_bars_do_not_change_past_state`、`test_incremental_equals_fresh_computation`） |
| 用未来数据计算 scaler / 特征 | 特征编码只消费 `ChanState` 快照，快照本身因果；scaler 阶段沿用 chronological split 约束（不在本次范围内） |

## 10. 测试策略

`tests/test_causal_chan.py`（13 个用例，合成确定性 zigzag K 线，无随机性）：

1. 基本缠论计算（笔/线段/中枢字段合法性）
2. Point-in-Time（as_of 边界严格性，`close_time == as_of` 被排除）
3. Future leakage（T1 状态在投喂后续 K 线后完全不变）
4. Incremental vs 截止 T 独立计算一致性
5. 未收盘 K 线排除
6. 多周期（1m/5m/15m/1h 全部可生成状态）
7. 序列化 round-trip（JSON + 键序稳定）
8. 缺失结构 mask（2 根 K 线、空数据均不崩溃）
9. Determinism（相同输入 → 完全相同输出）
10. Feature encoder（键序 / NaN 语义 / one-hot / 多周期前缀）

回归：原有 42 个测试（含 `tests/test_chan_engine.py` 对 `SimpleChanEngine` 的测试）必须继续通过 —— `SimpleChanEngine` 与既有接口未做任何改动。

## 11. 当前已知的缠论定义差异 / 限制

1. **无 4h 级别**：chan.py `KL_TYPE` 缺失 `K_240M`，4h 缠论状态被如实 mask（见 §6）。
2. **单级别独立计算**：v1 不使用 chan.py 的多级别联立（父子级别对齐），与「联立」派的缠论结论可能不同（见 §6）。
3. **背驰不单独暴露**：chan.py 开源 API 只在买卖点内部使用背驰判定，`divergence_*` 字段 v1 恒 masked；买卖点本身仍正常产出。
4. **买卖点无独立 is_sure**：以所属笔的 `is_sure` 为代理（字段名 `buy_sell_point_bi_is_sure` 如实标注）。
5. **K 线时间语义**：chan.py 日内级别的时间 = K 线**结束**时间，且对 A 股「00:00 视为当日收盘」有特殊处理 —— 我们统一用 UTC 墙钟 + `CTime(auto=False)` 屏蔽该行为（7×24 加密市场）。
6. **性能**：`trigger_step` 模式下线段/中枢在笔完成时全量重算，超长历史（如数十万根 1m K 线）首轮投喂较慢；引擎的增量缓存已避免重复投喂。
7. **买卖点消失**：chan.py 文档明确说明，未确定的买卖点随 K 线新增可能消失或移动 —— 这是缠论多义性的正常表现，也是本项目坚持 point-in-time 快照的原因。

## 12. chan.py 买卖点为什么不能作为 KOL 行为 Ground Truth

- 本项目的建模目标是**克隆 KOL 的真实交易行为**（Label = 该 KOL 实际发生的 `LONG` / `SHORT` 动作），不是预测缠论买卖点。
- chan.py 的买卖点是「某个特定缠论定义 + 特定参数」下的计算产物：不同的分型/笔/线段参数会给出不同结果，未确定买卖点还会消失/漂移。把它当标签会让模型学习一个单一、可变的规则系统，而不是 KOL 行为。
- KOL 的盈亏、错误开仓同样是行为数据的一部分：不按 PNL 删除/加权样本，不用未来 PNL/MFE/MAE 作为开仓时刻的 observation。
- 缠论买卖点在本项目中只作为**输入特征的一类**（ChanState → 特征），与 K 线几何特征并列，供模型自行学习与 KOL 行为之间的关联。
