#!/usr/bin/env python3
"""Initialize shop types (hardcoded + LLM-generated).

Usage:
    python init_shops.py              # Generate shops with LLM
    python init_shops.py --hardcoded-only  # Only use hardcoded shops
"""

import argparse
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from shop_generator import load_all_shops, list_shops


def main():
    parser = argparse.ArgumentParser(description="Initialize shop types for simulation")
    parser.add_argument(
        "--hardcoded-only",
        action="store_true",
        help="Skip LLM generation and use only hardcoded shops",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🏪 INITIALIZING SHOPS")
    print("=" * 80)

    if args.hardcoded_only:
        print("\n⚠️  Note: Using hardcoded shops only (skipping LLM generation)")
        print("   To generate new shops via LLM, run without --hardcoded-only\n")

    try:
        shops = load_all_shops()
        print(list_shops(shops))
        print(f"\n✅ Total shops available: {len(shops)}")
        print(f"📄 Shops saved to: shop_types.json\n")
    except Exception as e:
        print(f"\n❌ Error initializing shops: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
