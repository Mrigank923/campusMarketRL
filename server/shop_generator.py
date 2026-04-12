"""Shop type generation and management system.

Provides hardcoded shop types and LLM-generated shop types.
Supports one-time generation with caching to avoid repeated API calls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# Hardcoded shop types based on existing enums
HARDCODED_SHOPS = [
    {
        "name": "cafe",
        "display_name": "☕ Cafe",
        "description": "Coffee and quick snacks for students on the go",
        "base_demand": 0.70,
        "seasonality": ["free_time"],
        "inventory_items": ["coffee", "tea", "snacks", "pastries", "beverages"],
        "source": "hardcoded",
        "difficulty": "medium",
        "margin": 0.40,
    },
    {
        "name": "stationary",
        "display_name": "📚 Stationary",
        "description": "Notebooks, pens, and study supplies for academic work",
        "base_demand": 0.65,
        "seasonality": ["exam", "semester_start"],
        "inventory_items": ["notebooks", "pens", "pencils", "erasers", "books"],
        "source": "hardcoded",
        "difficulty": "hard",
        "margin": 0.35,
    },
    {
        "name": "food",
        "display_name": "🍜 Food",
        "description": "Full meals and food items for lunch and dinner",
        "base_demand": 0.75,
        "seasonality": ["lunch_time", "dinner_time"],
        "inventory_items": ["rice", "vegetables", "meals", "curry", "bread"],
        "source": "hardcoded",
        "difficulty": "easy",
        "margin": 0.45,
    },
    {
        "name": "tech",
        "display_name": "💻 Tech",
        "description": "Electronics, gadgets, and tech accessories",
        "base_demand": 0.55,
        "seasonality": ["semester_start", "festival"],
        "inventory_items": ["phones", "chargers", "headphones", "cables", "adapters"],
        "source": "hardcoded",
        "difficulty": "hard",
        "margin": 0.50,
    },
]


def generate_llm_shops(
    num_shops: int = 6,
    api_config: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Generate new shop types using LLM.
    
    Args:
        num_shops: Number of shops to generate
        api_config: Optional API configuration with base_url, model, api_key
        
    Returns:
        List of generated shop dictionaries
    """
    try:
        import requests
    except ImportError:
        print("⚠️  requests library not found. Install with: pip install requests")
        return []

    if api_config is None:
        api_config = _load_api_config()

    if not api_config:
        print("⚠️  No API credentials found. Skipping LLM shop generation.")
        return []

    system_prompt = """You are a creative business consultant designing shop concepts for an Indian college campus.
Generate realistic, diverse shop types that students would find useful.

Requirements:
- Each shop must have a distinct purpose
- Consider Indian college culture and needs
- Include seasonal demand patterns
- Realistic product inventory
- Profit margins 0.3-0.6

Return ONLY valid JSON array, no other text."""

    user_prompt = f"""Generate {num_shops} unique shop types for a college campus.

Return JSON array with this structure for each shop:
{{
  "name": "shop_name (lowercase, no spaces)",
  "display_name": "Emoji Name",
  "description": "2-3 line description of what this shop sells",
  "base_demand": 0.3-0.9 (float),
  "seasonality": ["season1", "season2"],
  "inventory_items": ["item1", "item2", "item3", "item4", "item5"],
  "difficulty": "easy|medium|hard",
  "margin": 0.3-0.6 (float)
}}

Examples of seasons: "exam", "festival", "monsoon", "summer", "free_time", "lunch_time"

IMPORTANT: Return ONLY the JSON array, no markdown, no explanation."""

    try:
        response = requests.post(
            f"{api_config['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {api_config['api_key']}"},
            json={
                "model": api_config["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 2000,
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return []

        content = response.json()["choices"][0]["message"]["content"].strip()

        # Try to extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        shops = json.loads(content)

        # Validate and add metadata
        validated_shops = []
        for shop in shops:
            if _validate_shop(shop):
                shop["source"] = "llm"
                validated_shops.append(shop)

        print(f"✅ Generated {len(validated_shops)}/{num_shops} shops via LLM")
        return validated_shops

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse LLM response as JSON: {e}")
        return []
    except (KeyError, IndexError) as e:
        print(f"❌ Unexpected API response format: {e}")
        return []


def _validate_shop(shop: dict[str, Any]) -> bool:
    """Validate shop dictionary has required fields."""
    required_fields = {
        "name",
        "display_name",
        "description",
        "base_demand",
        "seasonality",
        "inventory_items",
        "difficulty",
        "margin",
    }

    if not isinstance(shop, dict):
        return False

    missing = required_fields - set(shop.keys())
    if missing:
        print(f"⚠️  Shop missing fields: {missing}")
        return False

    # Validate types
    try:
        assert isinstance(shop["name"], str) and shop["name"].islower()
        assert isinstance(shop["description"], str) and len(shop["description"]) > 10
        assert 0.0 <= shop["base_demand"] <= 1.0
        assert isinstance(shop["seasonality"], list)
        assert isinstance(shop["inventory_items"], list) and len(shop["inventory_items"]) >= 3
        assert shop["difficulty"] in ["easy", "medium", "hard"]
        assert 0.3 <= shop["margin"] <= 0.6
        return True
    except AssertionError:
        return False


def _load_api_config() -> dict[str, str]:
    """Load API configuration from environment or .env file."""
    config = {}

    # Try .env file
    if os.path.exists(".env"):
        try:
            with open(".env") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        if key in ["HUGGINGFACE_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL"]:
                            config[key] = value
        except Exception:
            pass

    # Try environment variables
    api_key = config.get("HUGGINGFACE_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
    model = config.get("LLM_MODEL_ID") or os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    base_url = config.get("LLM_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://api-inference.huggingface.co/v1")

    if api_key:
        return {"api_key": api_key, "model": model, "base_url": base_url}

    return {}


def load_all_shops() -> list[dict[str, Any]]:
    """Load all shops (hardcoded + cached LLM or fresh generation)."""
    shop_types_file = "shop_types.json"

    # Load from cache if exists
    if os.path.exists(shop_types_file):
        try:
            with open(shop_types_file) as f:
                data = json.load(f)
                shops = data.get("shop_types", [])
                if shops:
                    print(f"✅ Loaded {len(shops)} shops from cache")
                    return shops
        except Exception as e:
            print(f"⚠️  Failed to load shop cache: {e}")

    # Generate fresh
    print("📝 Initializing shop types...")
    all_shops = list(HARDCODED_SHOPS)

    llm_shops = generate_llm_shops(num_shops=6)
    all_shops.extend(llm_shops)

    # Save to cache
    try:
        os.makedirs(os.path.dirname(shop_types_file) or ".", exist_ok=True)
        with open(shop_types_file, "w") as f:
            json.dump({"shop_types": all_shops, "count": len(all_shops)}, f, indent=2)
        print(f"💾 Saved {len(all_shops)} shops to {shop_types_file}")
    except Exception as e:
        print(f"⚠️  Failed to cache shops: {e}")

    return all_shops


def get_shop_by_index(shops: list[dict[str, Any]], index: int) -> Optional[dict[str, Any]]:
    """Get shop by 1-based index."""
    if 1 <= index <= len(shops):
        return shops[index - 1]
    return None


def get_shop_by_name(shops: list[dict[str, Any]], name: str) -> Optional[dict[str, Any]]:
    """Get shop by name."""
    name_lower = name.lower().strip()
    for shop in shops:
        if shop["name"].lower() == name_lower:
            return shop
    return None


def list_shops(shops: list[dict[str, Any]]) -> str:
    """Format shops list for display."""
    lines = ["\n📌 Available Shop Types:\n"]
    for i, shop in enumerate(shops, 1):
        source_tag = "💾" if shop.get("source") == "hardcoded" else "🤖"
        difficulty = shop.get("difficulty", "?").upper()
        lines.append(
            f"  {i}. {shop['display_name']:15} {source_tag}  {shop['description'][:50]:50} [{difficulty}]"
        )
    return "".join(lines)


if __name__ == "__main__":
    # Test: Generate and display shops
    shops = load_all_shops()
    print(list_shops(shops))
    print(f"\nTotal shops available: {len(shops)}")
