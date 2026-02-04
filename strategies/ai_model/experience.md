---
name: ai_model_strategy
description: LLM + prompt strategy that uses multi-factor information (fundamentals, technicals, news, macro) to produce trade decisions and reasoning.
tags: [strategy, AI, 大模型, multi-factor, 策引]
---

## Core rule
**AI model strategy** does not rely on a single indicator; it takes **multiple inputs** (fundamentals, technicals, news, macro) and uses a **large model + prompt** to output a decision (e.g. BUY/SELL/HOLD) and **reasoning**. Suited to complex, multi-factor decisions and personalized style (e.g. conservative, aggressive, value, growth).

## Inputs (from doc)
- **Fundamentals**: Financials, valuation, profitability, growth.
- **Technicals**: Price trend, indicators, volume.
- **News / sentiment**: Events, sentiment, narrative.
- **Macro**: Economy, policy, sector/cycle.

## Outputs
- **Decision**: BUY, SELL, HOLD (or similar).
- **Reasoning**: Short explanation of main factors and logic (for transparency and audit).

## Advantages
- **Multi-factor**: Can weigh many dimensions at once.
- **Adaptive**: Prompt and context can shift focus by regime or style.
- **Explainable**: Reasoning text supports review and compliance.

## Use cases
- Users who want AI to synthesize many data sources.
- Customization by risk/style (e.g. value vs growth, conservative vs aggressive).
- Research and decision support rather than fully automated execution (per 策引’s design).

## See also
- `basic_concepts`: AI/LLM strategy layer
- 策引 doc: [AI大模型策略](https://docs.myinvestpilot.com/docs/strategies/ai-model-strategy)
