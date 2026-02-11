"""Place bets on Polymarket from active.json."""

import argparse
import os

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs
from py_clob_client.constants import POLYGON

from polymarket_helpers.gamma import fetch_nba_events, find_market
from polymarket_helpers.matching import parse_matchup, event_matches_matchup, pick_matches_outcome
from polymarket_helpers.odds import format_price_comparison
from workflow.io import get_active_bets

POLYMARKET_HOST = "https://clob.polymarket.com"


def resolve_token_id(bet: dict, events: list[dict]) -> tuple[str, float] | None:
    """Find the CLOB token ID and price for a bet from Polymarket events."""
    try:
        away, home = parse_matchup(bet["matchup"])
    except ValueError:
        return None

    for event in events:
        if not event_matches_matchup(event.get("title", ""), away, home):
            continue

        market = find_market(event, bet["bet_type"], bet.get("line"))
        if not market:
            continue

        outcomes = market["outcomes"]
        prices = [float(p) for p in market["outcomePrices"]]
        token_ids = market["clobTokenIds"]

        for i, outcome in enumerate(outcomes):
            if pick_matches_outcome(bet["pick"], outcome):
                return token_ids[i], prices[i]

    return None


def create_clob_client(private_key: str, funder: str) -> ClobClient:
    """Create an authenticated ClobClient."""
    return ClobClient(
        host=POLYMARKET_HOST,
        chain_id=POLYGON,
        key=private_key,
        funder=funder,
    )


def place_bet(client: ClobClient, token_id: str, amount: float) -> dict:
    """Place a market buy order on Polymarket."""
    order_args = MarketOrderArgs(token_id=token_id, amount=amount, side="BUY")
    signed_order = client.create_market_order(order_args)
    return client.post_order(signed_order, orderType="FOK")


def run(date: str) -> None:
    """Load active bets for date, resolve markets, and place orders."""
    load_dotenv()

    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    funder = os.environ.get("POLYMARKET_FUNDER")
    if not private_key or not funder:
        print("Error: POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER must be set")
        return

    bets = [b for b in get_active_bets() if b["date"] == date]
    if not bets:
        print(f"No active bets for {date}")
        return

    print(f"Found {len(bets)} active bet(s) for {date}")

    client = create_clob_client(private_key, funder)

    events = fetch_nba_events(date)
    if not events:
        print("No Polymarket events found for this date")
        return

    print(f"Found {len(events)} Polymarket event(s)\n")

    placed = 0
    skipped = 0

    for bet in bets:
        label = f"{bet['matchup']} | {bet['bet_type']} {bet['pick']}"
        result = resolve_token_id(bet, events)

        if not result:
            print(f"SKIP: {label} -> no matching market")
            skipped += 1
            continue

        token_id, poly_price = result
        odds_price = bet.get("odds_price")
        if odds_price:
            print(f"  {format_price_comparison(odds_price, poly_price)}")

        amount = bet.get("amount", 0)
        if amount <= 0:
            print(f"SKIP: {label} -> no amount set")
            skipped += 1
            continue

        try:
            resp = place_bet(client, token_id, amount)
            print(f"OK:   {label} -> ${amount:.2f} placed")
            print(f"      Response: {resp}")
            placed += 1
        except Exception as e:
            print(f"FAIL: {label} -> {e}")
            skipped += 1

    print(f"\nDone: {placed} placed, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description="Place active bets on Polymarket")
    parser.add_argument("--date", required=True, help="Date to place bets for (YYYY-MM-DD)")
    args = parser.parse_args()
    run(args.date)


if __name__ == "__main__":
    main()
