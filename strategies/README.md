# Strategies knowledge

This directory holds **strategy knowledge** for the agent, in the same spirit as **skills**: each subdirectory is a named "strategy" with an `experience.md` that summarizes concepts, rules, and pointers. Content is aligned with [策引文档](https://docs.myinvestpilot.com/docs/strategies/basic-concepts) (MyInvestPilot).

## Structure

Each strategy has:

- **`experience.md`**: Frontmatter (`name`, `description`, `tags`), core rules, and "See also" links. Format matches the code **skills** under `skills/`.

## Index

| Subdir | Name | Description |
|--------|------|-------------|
| `basic_concepts/` | basic_concepts | Trading vs capital strategy, signal states (BUY/SELL/HOLD/EMPTY), three-layer framework (code / primitive / AI) |
| `buy_and_hold/` | buy_and_hold | Buy-and-hold for 定投; emits buy intent; capital strategy decides amount |
| `dual_moving_average/` | dual_moving_average | Golden cross / death cross; MA parameters; market behavior and improvements |
| `chandelier_exit/` | chandelier_exit | ATR-based trailing stop (long: highest − ATR×multiplier) |
| `target_weight/` | target_weight | Target weights and rebalancing |
| `momentum_rotation/` | momentum_rotation | Cross-asset momentum ranking and rotation (code strategy) |
| `macd/` | macd_strategy | MACD / signal crossover and typical parameters |
| `bollinger_bands/` | bollinger_bands | Bollinger Bands: mean reversion vs breakout |
| `primitive_strategy/` | primitive_strategy | Config-based, no-code strategies (indicator + signal primitives) |
| `ai_model/` | ai_model_strategy | LLM + prompt, multi-factor (fundamentals, technicals, news, macro) |
| `file_based/` | file_based_strategy | Strategy driven by external trade records (file) |

## Doc references

- **Base**: [交易策略基础](https://docs.myinvestpilot.com/docs/strategies/basic-concepts)
- **Code strategies**: [买入持有](https://docs.myinvestpilot.com/docs/strategies/buy-and-hold), [双均线](https://docs.myinvestpilot.com/docs/strategies/dual-moving-average), [目标权重](https://docs.myinvestpilot.com/docs/strategies/target-weight-strategy), [吊灯止损](https://docs.myinvestpilot.com/docs/strategies/chandelier-exit), [动量轮动](https://docs.myinvestpilot.com/docs/strategies/momentum-rotation), [MACD](https://docs.myinvestpilot.com/docs/strategies/macd-strategy), [布林带](https://docs.myinvestpilot.com/docs/strategies/bollinger-bands)
- **Other**: [原语策略](https://docs.myinvestpilot.com/docs/strategies/primitive-strategy), [AI大模型策略](https://docs.myinvestpilot.com/docs/strategies/ai-model-strategy), [交易记录策略](https://docs.myinvestpilot.com/docs/strategies/file-based-strategy)

## Relation to skills

- **Skills** (`skills/`): Implementation patterns (e.g. `backtest_ma_crossover`, `risk_metrics`, `adj_prices_and_returns`) and data/API usage.
- **Strategies** (`strategies/`): Strategy **concepts** and 策引’s strategy types—when to use which, signal semantics, parameters, and doc links. Use strategies for reasoning about "what strategy fits"; use skills for "how to implement" in code.
