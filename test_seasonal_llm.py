#!/usr/bin/env python3
"""
Test script with seasonal awareness and LLM-influenced trends.

Features:
- User selects starting month and simulation duration
- Trends follow Indian seasons (Nov-Feb winter, etc.)
- LLM's performance influences market trends
- Detailed per-step metrics and seasonal context

Usage:
    python test_seasonal_llm.py --start-month January --days 30
    python test_seasonal_llm.py --start-month July --days 60
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Import core functionality
from campus_market_env.enums import PhaseEnum, ShopTypeEnum, TrendTypeEnum
from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv

# Import seasonal trend model functions
# Since this file is in root, we can use sys.path manipulation
sys.path.insert(0, str(Path(__file__).parent / "server"))
try:
    from seasonal_trend_model import (
        INDIAN_SEASONS,
        MONTH_TO_NUMBER,
        MONTH_NAMES,
        adjust_trend_for_llm_performance,
        get_base_seasonal_trend,
        get_seasonal_trend,
        get_trend_description,
        get_trend_multiplier,
        get_season_for_month,
    )
    from shop_generator import load_all_shops, list_shops, get_shop_by_index
finally:
    sys.path.pop(0)


# Load environment variables
load_dotenv()
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def build_user_prompt(step: int, observation: dict, season_info: str) -> str:
    """Build a detailed prompt with seasonal context."""
    prompt = f"""You are a campus market manager. It's {season_info}.

Current Market State (Step {step}):
- Day: {observation['day']}
- Phase: {observation['phase']}
- Customer Satisfaction: {observation['customer_satisfaction']:.1%}
- Awareness: {observation['awareness']:.1%}
- Market Sentiment: {observation['market_sentiment']:.1%}
- Competition Pressure: {observation['competitor_pressure']:.1%}
- Trend: {observation.get('trend_description', 'Normal')}
- Inventory Level: {observation['inventory_level']:.1%}
- Monthly Budget: ${observation['monthly_budget']:.2f}
- Recent Revenue: ${observation['revenue']:.2f}
- Shop Traffic: {observation['shop_traffic']} students

Your decision (JSON format):
{{
    "price_adjustment": <float between -1.0 and 1.0>,
    "marketing_spend": <float >= 0.0>,
    "restock_amount": <integer >= 0>,
    "reasoning": "<brief explanation of your decision>"
}}

Focus on: maintaining satisfaction, managing budget, adapting to seasonal trends."""
    return prompt


def call_llm_for_action(client: OpenAI, prompt: str, step: int) -> dict | None:
    """Call LLM to get action decision."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=300,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                action_dict = json.loads(json_str)
                return action_dict
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    except Exception as e:
        print(f"⚠️  LLM API Error at step {step}: {str(e)}")
        return None


def get_heuristic_action(observation: dict) -> dict:
    """Fallback heuristic when LLM fails."""
    # Simple strategy: maintain inventory, spend on marketing, balance price
    return {
        "price_adjustment": 0.0,
        "marketing_spend": min(100.0, observation["monthly_budget"] * 0.1),
        "restock_amount": max(0, int(20 * (1.0 - observation["inventory_level"]))),
        "product_focus": "cafe",
        "reasoning": "Heuristic fallback: balanced strategy",
    }


def run_seasonal_episode(
    start_month: str,
    num_days: int,
    initial_budget: float,
    use_llm: bool = True,
    shop_type: str = "cafe",
) -> tuple[float, dict]:
    """
    Run a single episode with seasonal awareness.
    
    Args:
        start_month: Starting month name ("January", "July", etc.)
        num_days: Number of days to simulate
        initial_budget: Starting budget in dollars
        use_llm: Whether to use LLM for decisions
        shop_type: Type of shop to simulate ("cafe", "food", "tech", "stationary", or custom)
    
    Returns:
        (total_reward, episode_stats)
    """
    
    # Validate start month
    if start_month not in MONTH_TO_NUMBER:
        raise ValueError(f"Invalid month: {start_month}. Must be one of {list(MONTH_TO_NUMBER.keys())}")
    
    start_month_num = MONTH_TO_NUMBER[start_month]
    season_label, season_name, months = INDIAN_SEASONS[start_month]
    
    print("\n" + "=" * 80)
    print(f"🌍 SEASONAL MARKET SIMULATION")
    print("=" * 80)
    print(f"Shop Type: {shop_type.upper()}")
    print(f"Starting Month: {start_month} (Season: {season_name.upper()})")
    print(f"Season Months: {season_label}")
    print(f"Simulation Duration: {num_days} days")
    print(f"Starting Budget: ${initial_budget:,.2f}")
    print(f"LLM Enabled: {'✅ YES' if use_llm else '❌ NO (Heuristic)'}")
    print("=" * 80 + "\n")
    
    # Create environment
    env = CampusMarketEnv(seed=12345)
    observation = env.reset(seed=12345, shop_type=shop_type)
    
    client = None
    if use_llm and API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    total_reward = 0.0
    total_steps = 0
    marketing_total = 0.0
    rewards_list = []
    satisfactions = []
    
    # Convert observation to dict if needed
    if hasattr(observation, 'model_dump'):
        obs_dict = observation.model_dump()
    else:
        obs_dict = dict(observation)
    
    step = 0
    while step < num_days and total_steps < 270:  # 270 max steps = 90 days × 3 phases
        step += 1
        
        # Calculate current month and season
        days_per_month = 30
        month_offset = (step - 1) // days_per_month
        current_month = ((start_month_num - 1 + month_offset) % 12) + 1
        current_month_name = MONTH_NAMES[current_month]
        current_season = get_season_for_month(current_month)
        
        # Get seasonal trend
        base_trend = get_seasonal_trend(start_month_num, step, seed=12345)
        
        # Add trend description to observation
        obs_dict['trend_description'] = get_trend_description(base_trend)
        trend_multiplier = get_trend_multiplier(base_trend)
        
        # Build season info for LLM
        season_info = f"{current_month_name} ({current_season.capitalize()})"
        
        # Get LLM action
        if use_llm and client:
            prompt = build_user_prompt(step, obs_dict, season_info)
            action_dict = call_llm_for_action(client, prompt, step)
            
            if action_dict is None:
                action_dict = get_heuristic_action(obs_dict)
        else:
            action_dict = get_heuristic_action(obs_dict)
        
        # Adjust trend based on LLM performance
        llm_prices_high = action_dict.get("price_adjustment", 0.0) > 0.3
        inventory_status = (
            "low" if obs_dict["inventory_level"] < 0.3 
            else "high" if obs_dict["inventory_level"] > 0.7 
            else "balanced"
        )
        
        adjusted_trend = adjust_trend_for_llm_performance(
            base_trend=base_trend,
            customer_satisfaction=obs_dict["customer_satisfaction"],
            market_sentiment=obs_dict["market_sentiment"],
            llm_marketing_spend=action_dict.get("marketing_spend", 0.0),
            llm_prices_high=llm_prices_high,
            inventory_status=inventory_status,
        )
        
        # Check if trend changed
        trend_changed = adjusted_trend != base_trend
        
        # Create action
        try:
            action = CampusMarketAction(
                price_adjustment=max(-1.0, min(1.0, action_dict.get("price_adjustment", 0.0))),
                marketing_spend=max(0.0, action_dict.get("marketing_spend", 0.0)),
                restock_amount=max(0, int(action_dict.get("restock_amount", 0))),
            )
        except Exception as e:
            print(f"⚠️  Action validation error at step {step}: {e}")
            action = CampusMarketAction(
                price_adjustment=0.0,
                marketing_spend=0.0,
                restock_amount=0,
            )
        
        # Step environment
        observation = env.step(action)
        reward = observation.reward if hasattr(observation, 'reward') else 0.0
        done = observation.done if hasattr(observation, 'done') else False
        truncated = done
        
        # Convert observation
        if hasattr(observation, 'model_dump'):
            obs_dict = observation.model_dump()
        else:
            obs_dict = dict(observation)
        
        total_reward += reward
        total_steps += 1
        marketing_total += action.marketing_spend
        rewards_list.append(reward)
        satisfactions.append(obs_dict.get("customer_satisfaction", 0.0))
        
        # Print step summary
        if step % 10 == 1 or step == 1:
            print(f"\n{'─' * 80}")
            print(f"📅 Day {step:2d} | {current_month_name} ({current_season.upper()}) | "
                  f"Phase: {obs_dict.get('phase', 'unknown')}")
            print(f"{'─' * 80}")
        
        trend_str = f"{get_trend_description(adjusted_trend)}"
        if trend_changed:
            trend_str += f" (shifted from {base_trend.value})"
        
        print(f"  {trend_str}")
        print(f"  💰 Revenue: ${obs_dict.get('revenue', 0):.2f} | "
              f"Satisfaction: {obs_dict.get('customer_satisfaction', 0):.1%} | "
              f"Reward: {reward:.2f}")
        print(f"  📊 Awareness: {obs_dict.get('awareness', 0):.1%} | "
              f"Inventory: {obs_dict.get('inventory_level', 0):.1%} | "
              f"Budget: ${obs_dict.get('monthly_budget', 0):.2f}")
        print(f"  🎯 Action: Price {action.price_adjustment:+.2f} | "
              f"Marketing: ${action.marketing_spend:.1f} | "
              f"Restock: {action.restock_amount}")
        
        if done or truncated:
            break
    
    env.close()
    
    # Calculate statistics
    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    avg_satisfaction = sum(satisfactions) / len(satisfactions) if satisfactions else 0
    
    stats = {
        "total_reward": total_reward,
        "avg_reward_per_step": avg_reward,
        "total_steps": total_steps,
        "total_marketing_spend": marketing_total,
        "final_satisfaction": obs_dict.get("customer_satisfaction", 0),
        "avg_satisfaction": avg_satisfaction,
        "final_inventory": obs_dict.get("inventory_level", 0),
        "final_budget": obs_dict.get("monthly_budget", 0),
        "start_month": start_month,
        "season": season_name,
        "initial_budget": initial_budget,
    }
    
    return total_reward, stats


def interactive_setup() -> tuple[str, int, float, str]:
    """
    Interactively ask user for setup parameters.
    Returns: (start_month, num_days, initial_budget, shop_type)
    """
    print("\n" + "=" * 80)
    print("🎓 CAMPUS MARKET SIMULATION - SETUP")
    print("=" * 80)
    
    # Load available shops
    shops = load_all_shops()
    
    # Shop selection
    print(list_shops(shops))
    
    while True:
        try:
            choice = int(input("\nSelect shop (1-{}): ".format(len(shops))))
            shop = get_shop_by_index(shops, choice)
            if shop:
                shop_type = shop["name"]
                print(f"\n✅ Selected: {shop['display_name']} ({shop['description'][:50]}...)")
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(shops)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Month selection
    print("\n📅 SELECT STARTING MONTH:")
    months = list(MONTH_TO_NUMBER.keys())
    for i, month in enumerate(months, 1):
        print(f"   {i:2d}. {month}")
    
    while True:
        try:
            choice = int(input("\nEnter month number (1-12): "))
            if 1 <= choice <= 12:
                start_month = months[choice - 1]
                break
            else:
                print("❌ Please enter a number between 1 and 12")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Days selection
    print(f"\n✅ Selected: {start_month}")
    print("\n📆 HOW MANY DAYS TO SIMULATE?")
    print("   • 10 days = Quick test (3 min)")
    print("   • 30 days = Good test (10 min)")
    print("   • 90 days = Full quarter (30 min)")
    
    while True:
        try:
            days = int(input("\nEnter number of days (1-90): "))
            if 1 <= days <= 90:
                break
            else:
                print("❌ Please enter a number between 1 and 90")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Budget selection
    print(f"\n✅ Selected: {days} days")
    print("\n💰 STARTING BUDGET FOR YOUR SHOP:")
    print("   • $5000  = Limited budget (hard challenge)")
    print("   • $10000 = Moderate budget (balanced)")
    print("   • $15000 = Good budget (easier)")
    print("   • $20000 = Plenty of budget (easiest)")
    
    while True:
        try:
            budget = float(input("\nEnter starting budget in dollars (e.g., 10000): "))
            if budget > 0:
                break
            else:
                print("❌ Budget must be greater than 0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"\n✅ Selected: ${budget:.2f}")
    
    return start_month, days, budget, shop_type


def main():
    parser = argparse.ArgumentParser(
        description="Test campus market environment with seasonal awareness and LLM-influenced trends."
    )
    parser.add_argument(
        "--shop",
        type=int,
        default=None,
        help="Shop type to simulate (by index, 1-based)",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        default=None,
        choices=list(MONTH_TO_NUMBER.keys()),
        help="Starting month for simulation (if not provided, will ask interactively)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to simulate (if not provided, will ask interactively)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Starting budget in dollars (if not provided, will ask interactively)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM and use heuristic strategy instead",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use default values without interactive setup",
    )
    
    args = parser.parse_args()
    
    # Get parameters from command line or ask interactively
    if args.auto or (args.start_month and args.days and args.budget):
        # Use provided values
        start_month = args.start_month or "January"
        num_days = args.days or 30
        initial_budget = args.budget or 10000.0
        
        # Get shop type
        if args.shop:
            shops = load_all_shops()
            shop_info = get_shop_by_index(shops, args.shop)
            if shop_info:
                shop_type = shop_info["name"]
            else:
                print(f"❌ Invalid shop index: {args.shop}")
                sys.exit(1)
        else:
            shop_type = "cafe"  # Default
    else:
        # Ask user interactively
        start_month, num_days, initial_budget, shop_type = interactive_setup()
    
    try:
        total_reward, stats = run_seasonal_episode(
            start_month=start_month,
            num_days=num_days,
            initial_budget=initial_budget,
            use_llm=not args.no_llm,
            shop_type=shop_type,
        )
        
        # Print final summary
        print("\n" + "=" * 80)
        print("📊 EPISODE SUMMARY")
        print("=" * 80)
        print(f"Shop Type: {shop_type.upper()}")
        print(f"Season: {stats['season'].upper()} ({stats['start_month']})")
        print(f"Starting Budget: ${stats['initial_budget']:,.2f}")
        print(f"Final Budget: ${stats['final_budget']:.2f}")
        print(f"Total Marketing Spent: ${stats['total_marketing_spend']:.2f}")
        print(f"\nTotal Reward: {stats['total_reward']:.2f}")
        print(f"Avg Reward/Step: {stats['avg_reward_per_step']:.2f}")
        print(f"Total Steps: {stats['total_steps']}")
        print(f"Total Marketing: ${stats['total_marketing_spend']:.2f}")
        print(f"Final Satisfaction: {stats['final_satisfaction']:.1%}")
        print(f"Avg Satisfaction: {stats['avg_satisfaction']:.1%}")
        print(f"Final Inventory: {stats['final_inventory']:.1%}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
