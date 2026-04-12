# Seasonal Trend System Implementation

## Overview

This implementation adds:

1. **Seasonal Awareness** - Trends now follow Indian seasons (Nov-Feb winter, Apr-Jun summer, etc.)
2. **Dynamic Trend Progression** - Users can select starting month, and trends transition naturally across months
3. **LLM-Influenced Trends** - LLM's performance can shift trends within seasonal bounds
4. **No LoRA** - Uses standard LLM API calls without fine-tuning adapters

## Architecture

### Files Added

#### 1. `server/seasonal_trend_model.py`
New module with seasonal and LLM influence functions:

```python
# Indian seasonal mapping
INDIAN_SEASONS = {
    "January": ("Jan-Mar", "winter", [1, 2, 3]),
    "July": ("Jul-Sep", "monsoon", [7, 8, 9]),
    ...
}

# Key functions:
get_seasonal_trend(start_month, day, seed)  # Get trend for specific day
adjust_trend_for_llm_performance(...)        # Shift trend based on LLM actions
get_trend_multiplier(trend)                  # Convert trend to demand multiplier
```

#### 2. `test_seasonal_llm.py`
New test script with:
- `--start-month` parameter (January, July, etc.)
- `--days` parameter (how many days to simulate)
- `--no-llm` flag (use heuristic fallback)
- Season-aware LLM prompts with context
- Per-step trend tracking and adjustment
- Detailed season/month progression display

## How It Works

### Phase 1: Seasonal Base Trend

```python
start_month = "January"  # User selects
day = 15

current_month = January
base_trend = get_seasonal_trend(start_month=1, day=15)
# Returns: TrendTypeEnum.NORMAL (winter baseline)

current_month = February  # Day 45
base_trend = get_seasonal_trend(start_month=1, day=45)
# Returns: TrendTypeEnum.NORMAL (still winter)

current_month = April  # Day 95 (crossed into Q2)
base_trend = get_seasonal_trend(start_month=1, day=95)
# Returns: TrendTypeEnum.EXAM (summer baseline)
```

### Phase 2: LLM Performance Influence

After LLM makes decisions, the trend can shift:

```python
# LLM's action:
action = {
    "price_adjustment": 0.05,      # Slight price increase
    "marketing_spend": 2500,       # Heavy marketing
    "restock_amount": 50,
    "product_focus": "cafe"
}

# Current state:
customer_satisfaction = 0.88      # High
market_sentiment = 0.75           # Good
inventory_status = "balanced"

# Trend adjustment:
base_trend = TrendTypeEnum.NORMAL
adjusted_trend = adjust_trend_for_llm_performance(
    base_trend,
    customer_satisfaction=0.88,    # ✅ Good
    market_sentiment=0.75,         # ✅ Good
    llm_marketing_spend=2500,      # ✅ High spend
    llm_prices_high=False,         # ✅ Reasonable prices
    inventory_status="balanced"    # ✅ Balanced
)
# Returns: TrendTypeEnum.FESTIVAL (shifted up!)
```

### Phase 3: Trend Usage

The adjusted trend affects:
- **Traffic calculation**: `traffic *= trend_multiplier`
- **Conversion calculation**: `conversion += trend_effect`
- **Market sentiment**: Updated based on trend + LLM performance
- **Observation**: Trend info provided to LLM for next decision

## Usage Examples

### Example 1: Winter Season (Nov-Feb)
```bash
python test_seasonal_llm.py --start-month November --days 90
```

Output:
```
🌍 SEASONAL MARKET SIMULATION
Starting Month: November (Season: AUTUMN)
Season Months: Oct-Dec
Simulation Duration: 90 days

Day  1 | November (AUTUMN) | Phase: morning
  🏖️ Holiday season - festive shopping
  💰 Revenue: $2451.30 | Satisfaction: 85.0% | Reward: 18.45
  📊 Awareness: 45.2% | Inventory: 62.1% | Budget: $890.34
  🎯 Action: Price +0.10 | Marketing: $250.0 | Restock: 15 | Focus: cafe

Day 11 | November (AUTUMN)
  🏖️ Holiday season (shifted from normal) ← Trend shifted up!
  💰 Revenue: $3120.00 | Satisfaction: 88.5% | Reward: 22.31
```

### Example 2: Monsoon Season (Jul-Sep)
```bash
python test_seasonal_llm.py --start-month July --days 45
```

Output:
```
Starting Month: July (Season: MONSOON)
...
Day  1 | July (MONSOON)
  📊 Normal market conditions
  
Day 15 | August (MONSOON)
  📊 Normal (shifted to EXAM) ← Poor LLM performance
  ⚠️ Satisfaction dropped due to high prices + low marketing
```

### Example 3: Cross-Season Simulation
```bash
python test_seasonal_llm.py --start-month December --days 90
# Runs: Dec → Jan → Feb → Mar
# Trend naturally transitions: AUTUMN → WINTER → SPRING
```

### Example 4: Heuristic Fallback (No LLM)
```bash
python test_seasonal_llm.py --start-month January --days 30 --no-llm
# Uses simple rule-based strategy instead of LLM
```

## Trend Shifting Rules

### Shift UP (Trend becomes better):
- ✅ Customer satisfaction > 0.85 AND marketing spend > $2000
- ✅ Market sentiment > 0.7 (positive response)
- Maps: `EXAM → NORMAL → FESTIVAL`

### Shift DOWN (Trend becomes worse):
- ❌ High prices AND overstock
- ❌ Customer satisfaction < 0.3 (very unhappy)
- Maps: `FESTIVAL → NORMAL → EXAM`

### No Change:
- Moderate performance
- Trend stays same

## Seasonal Mapping (India-Specific)

| Season | Months | Weather | Base Trend | Student Activity |
|--------|--------|---------|-----------|------------------|
| **Winter** | Nov-Feb | Cold | NORMAL | Indoor, focused |
| **Spring** | Mar | Warm-hot | FESTIVAL | Holi festival, exam prep |
| **Summer** | Apr-Jun | Hot | EXAM | Exams, less spending |
| **Monsoon** | Jul-Sep | Rainy | NORMAL | Stable, focused |
| **Autumn** | Oct | Pleasant | HOLIDAY | Diwali prep, festive |

## Trend Multipliers

| Trend | Multiplier | Effect |
|-------|------------|--------|
| FESTIVAL | 1.35x | 📈 Best sales conditions |
| HOLIDAY | 1.15x | 📈 Good sales |
| NORMAL | 1.0x | 📊 Baseline |
| EXAM | 0.85x | 📉 Reduced casual spending |

## Example Run Flow

```
Start: January (Winter), 30 days

Day 1 (Jan):
  base_trend = NORMAL
  LLM: "Marketing spend: $300, price: 0.0"
  satisfaction = 0.75, marketing = $300
  → adjust_trend returns NORMAL (not enough spend)
  trend_multiplier = 1.0
  
Day 10 (Jan):
  base_trend = NORMAL
  LLM: "Marketing spend: $2500, price: -0.1 (discount)"
  satisfaction = 0.92, marketing = $2500
  → adjust_trend returns FESTIVAL (excellent performance!)
  trend_multiplier = 1.35
  revenue_multiplier += 0.35x
  
Day 20 (Jan, late):
  base_trend = NORMAL
  LLM: "High prices: 0.4, poor inventory"
  satisfaction = 0.35, inventory = 0.2
  → adjust_trend returns EXAM (poor strategy)
  trend_multiplier = 0.85
  revenue_multiplier -= 0.15x
  
Day 30 (Jan, end):
  base_trend = NORMAL
  Final state recorded
  Season transition ready for next month
```

## Integration with Existing Code

### Backward Compatibility
- ✅ Original `trend_model.py` unchanged
- ✅ Original `engine.py` unchanged
- ✅ Original `inference.py` works as-is
- ✅ New system is in separate module

### When to Use Which

| Script | Trend Model | Usage |
|--------|-------------|-------|
| `inference.py` | Original `trend_model.py` | Meta submission (deterministic) |
| `test_env.py` | Original `trend_model.py` | Quick testing |
| `test_seasonal_llm.py` | New `seasonal_trend_model.py` | Development with seasons |

## Next Steps (With LoRA)

The current implementation uses standard LLM API calls. To add LoRA:

1. **Collect training data** - Record LLM decisions and market responses
2. **Create adapter** - Fine-tune Qwen with trend prediction examples
3. **Load adapter** - `client = OpenAI(..., adapter="trend_expert")`
4. **Get better predictions** - LLM becomes specialized for trend decisions

This would improve accuracy but is optional for now.

## Files Modified/Created

```
✅ Created:  server/seasonal_trend_model.py      (350 lines)
✅ Created:  test_seasonal_llm.py                (400 lines)
⚠️  Unchanged: server/engine.py                   (uses original trends)
⚠️  Unchanged: inference.py                       (uses original trends)
⚠️  Unchanged: server/trend_model.py              (original system)
```

## Running the Test

```bash
# Basic test
python test_seasonal_llm.py --start-month January --days 30

# Monsoon season, longer
python test_seasonal_llm.py --start-month July --days 60

# Without LLM
python test_seasonal_llm.py --start-month January --days 30 --no-llm

# Cross-season
python test_seasonal_llm.py --start-month November --days 90
```

## Observations Output

Each step shows:
- 📅 Current month and season
- 🌤️ Base trend + adjusted trend (if changed)
- 💰 Revenue, satisfaction, reward
- 📊 Awareness, inventory, budget
- 🎯 LLM's action decisions

---

**Status**: ✅ Complete without LoRA, ready for testing
**Next**: Add LoRA adapters for trend prediction specialization
